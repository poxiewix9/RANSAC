//
//  VisionPipeline.swift
//  SpatialMap
//
//  Phase 2 — turns raw camera frames into sparse 2D feature points.
//
//  PIPELINE PER FRAME (runs on CameraManager's background videoQueue):
//    1. THROTTLE   — most frames are dropped; we process ~10–15 FPS so we don't
//                    saturate the Multipeer link or pin the CPU/thermals.
//    2. DETECT     — VNDetectContoursRequest finds high-contrast edge contours
//                    (desk edges, laptop, mug rims…) on the Neural Engine/GPU.
//    3. SAMPLE     — contours are continuous paths; epipolar geometry needs
//                    discrete points, so we subsample contour vertices and cap
//                    the total at 200 points/frame to keep payloads tiny.
//    4. DISPATCH   — pack points into a FeaturePayload and stream to the peer,
//                    and publish the same points to the UI for a live overlay.
//
//  CONCURRENCY
//  -----------
//  Like CameraManager, this type is NOT @MainActor: `process(...)` is invoked
//  on a background serial queue once per frame. All throttle/counter state is
//  confined to that queue (frames arrive serially). Only the @Published
//  `localKeypoints` is hopped to the main thread, and the network send is
//  hopped to MultipeerManager's main actor. Marked @unchecked Sendable because
//  we manually uphold that confinement.
//

import Foundation
import Vision
import CoreVideo
import CoreMedia
import QuartzCore
import simd

final class VisionPipeline: ObservableObject, @unchecked Sendable {

    // MARK: Published UI state

    /// Normalized feature coordinates (Vision convention: origin BOTTOM-LEFT,
    /// range 0...1) for the SwiftUI scatter overlay. The view inverts Y when
    /// mapping to screen space.
    @Published private(set) var localKeypoints: [CGPoint] = []

    /// Rolling count of frames we actually processed (post-throttle). Useful
    /// for confirming the effective FPS during bring-up.
    @Published private(set) var processedFrameCount: Int = 0

    // MARK: Dependencies

    /// We push extracted features to the peer through this. It is @MainActor,
    /// so all calls into it are hopped to the main actor.
    private let multipeerManager: MultipeerManager

    init(multipeerManager: MultipeerManager) {
        self.multipeerManager = multipeerManager
    }

    // MARK: Tuning knobs

    /// Target processing rate. 1/12s ≈ 12 FPS — comfortably inside the 10–15
    /// FPS band that keeps the network and thermals happy.
    private let targetInterval: CFTimeInterval = 1.0 / 12.0

    /// Hard cap on points streamed per frame. Lower than Phase 2's 200 because
    /// each point now carries a 121-float patch descriptor (~0.5KB), so this
    /// keeps payloads in a sane range for the Multipeer link.
    private let maxPointsPerFrame = 150

    /// Vision processes a downscaled copy at this longest-edge size. Smaller =
    /// faster + fewer, cleaner contours. Detail is plenty for sparse features.
    private let maxImageDimension = 512

    /// Half-width of the square patch descriptor. radius 5 → an 11x11 window.
    private let patchRadius = 5

    /// Side length of the patch (2*radius + 1).
    private var patchSide: Int { patchRadius * 2 + 1 }

    // MARK: Background-confined state (videoQueue only)

    /// Wall-clock time of the last processed frame, for throttling.
    private var lastProcessTime: CFTimeInterval = 0

    /// Monotonic frame id we attach to outgoing payloads.
    private var frameID: UInt64 = 0

    /// Reusable request — configuring it once avoids per-frame allocation.
    private lazy var contoursRequest: VNDetectContoursRequest = {
        let request = VNDetectContoursRequest()
        // Crank contrast so only strong, stable edges survive — fewer noisy
        // contours means fewer false matches for RANSAC to reject in Phase 3.
        request.contrastAdjustment = 2.0
        // Detect both polarities so we catch light-on-dark and dark-on-light.
        request.detectsDarkOnLight = true
        // Downscale internally for speed.
        request.maximumImageDimension = maxImageDimension
        return request
    }()

    // MARK: Frame entry point (called on background videoQueue)

    /// Process one camera frame. Cheap-rejects most frames via the throttle.
    func process(pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        // --- 1. THROTTLE -----------------------------------------------------
        let now = CACurrentMediaTime()
        guard now - lastProcessTime >= targetInterval else { return }
        lastProcessTime = now

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // --- 2. DETECT -------------------------------------------------------
        // Buffers already arrive portrait-oriented (CameraManager rotates the
        // capture connection), so `.up` is correct here.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        do {
            try handler.perform([contoursRequest])
        } catch {
            return
        }

        guard let observation = contoursRequest.results?.first as? VNContoursObservation else {
            return
        }

        // --- 3. SAMPLE -------------------------------------------------------
        let points = sampleContourPoints(from: observation)

        // --- 3b. DESCRIBE ----------------------------------------------------
        // Attach an illumination-invariant patch descriptor to each point so
        // the peer can match by appearance (Phase 4). Points too close to the
        // image edge or sitting on a flat/featureless patch are dropped.
        let (featurePoints, uiPoints) = extractDescriptors(from: pixelBuffer,
                                                           at: points,
                                                           width: width,
                                                           height: height)

        // --- 4. DISPATCH -----------------------------------------------------
        frameID &+= 1
        dispatch(featurePoints: featurePoints,
                 uiPoints: uiPoints,
                 frameID: frameID,
                 width: width,
                 height: height)
    }

    // MARK: Patch descriptors

    /// For each sampled point, reads an 11x11 grayscale patch from the pixel
    /// buffer and normalizes it (zero mean, unit std) for lighting invariance.
    /// Returns the surviving FeaturePoints (with descriptors) and the matching
    /// CGPoints for the UI overlay, kept in lockstep.
    ///
    /// The capture format is 32BGRA (configured in CameraManager), so we derive
    /// luma per pixel as a weighted sum of the B/G/R bytes.
    private func extractDescriptors(from pixelBuffer: CVPixelBuffer,
                                    at points: [CGPoint],
                                    width: Int,
                                    height: Int) -> ([FeaturePoint], [CGPoint]) {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            // Unexpected format: fall back to descriptor-less points.
            let fallback = points.map { FeaturePoint(x: Float($0.x), y: Float($0.y)) }
            return (fallback, points)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return ([], [])
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        var featurePoints: [FeaturePoint] = []
        var uiPoints: [CGPoint] = []
        featurePoints.reserveCapacity(points.count)
        uiPoints.reserveCapacity(points.count)

        for p in points {
            // Vision coords are normalized BOTTOM-LEFT; pixel rows are TOP-LEFT,
            // so invert Y when indexing into the buffer.
            let px = Int(Float(p.x) * Float(width))
            let py = Int((1 - Float(p.y)) * Float(height))

            if let descriptor = patchDescriptor(ptr,
                                                bytesPerRow: bytesPerRow,
                                                width: width,
                                                height: height,
                                                cx: px,
                                                cy: py) {
                featurePoints.append(FeaturePoint(x: Float(p.x),
                                                  y: Float(p.y),
                                                  descriptor: descriptor))
                uiPoints.append(p)
            }
        }
        return (featurePoints, uiPoints)
    }

    /// Extracts and normalizes a single patch. Returns nil if the patch would
    /// fall outside the image, or if it's too flat to be discriminative.
    private func patchDescriptor(_ ptr: UnsafeMutablePointer<UInt8>,
                                 bytesPerRow: Int,
                                 width: Int,
                                 height: Int,
                                 cx: Int,
                                 cy: Int) -> [Float]? {
        let r = patchRadius
        // Reject points whose window spills past the image border.
        guard cx - r >= 0, cy - r >= 0, cx + r < width, cy + r < height else {
            return nil
        }

        var patch = [Float](repeating: 0, count: patchSide * patchSide)
        var i = 0
        for dy in -r...r {
            let rowBase = (cy + dy) * bytesPerRow
            for dx in -r...r {
                let off = rowBase + (cx + dx) * 4 // BGRA → 4 bytes/pixel
                let b = Float(ptr[off + 0])
                let g = Float(ptr[off + 1])
                let red = Float(ptr[off + 2])
                // Rec.601 luma weights.
                patch[i] = 0.114 * b + 0.587 * g + 0.299 * red
                i += 1
            }
        }

        // Normalize: subtract mean, divide by standard deviation.
        let count = Float(patch.count)
        let mean = patch.reduce(0, +) / count
        var variance: Float = 0
        for v in patch {
            let d = v - mean
            variance += d * d
        }
        variance /= count
        let std = sqrt(variance)
        guard std > 1e-5 else { return nil } // flat patch — not discriminative

        for j in 0..<patch.count {
            patch[j] = (patch[j] - mean) / std
        }
        return patch
    }

    // MARK: Sampling

    /// Flattens every top-level contour's vertices into a single point list,
    /// then evenly subsamples it down to `maxPointsPerFrame`.
    ///
    /// We read each contour's `normalizedPoints` (its discrete vertices in the
    /// 0...1, bottom-left space) rather than walking the CGPath manually — these
    /// ARE the path's vertices and are exactly the discrete points we want.
    private func sampleContourPoints(from observation: VNContoursObservation) -> [CGPoint] {
        var all: [CGPoint] = []
        all.reserveCapacity(observation.contourCount)

        for contour in observation.topLevelContours {
            // `normalizedPoints` is a buffer of simd_float2 vertices.
            for p in contour.normalizedPoints {
                all.append(CGPoint(x: CGFloat(p.x), y: CGFloat(p.y)))
            }
        }

        guard all.count > maxPointsPerFrame else { return all }

        // Even stride keeps points spread across the whole scene instead of
        // clustering on the first few contours.
        let stride = all.count / maxPointsPerFrame
        var sampled: [CGPoint] = []
        sampled.reserveCapacity(maxPointsPerFrame)
        var i = 0
        while i < all.count && sampled.count < maxPointsPerFrame {
            sampled.append(all[i])
            i += stride
        }
        return sampled
    }

    // MARK: Dispatch (network + UI)

    private func dispatch(featurePoints: [FeaturePoint],
                          uiPoints: [CGPoint],
                          frameID: UInt64,
                          width: Int,
                          height: Int) {
        // --- Network: build payload and send via the (main-actor) manager ---
        // Each FeaturePoint now carries a normalized 121-float patch descriptor
        // that the peer's GeometrySolver matches with SSD + Lowe's ratio test.
        let senderName = multipeerManager.localDisplayName // nonisolated read
        let payload = FeaturePayload(
            frameID: frameID,
            senderName: senderName,
            imageWidth: width,
            imageHeight: height,
            points: featurePoints
        )

        // MultipeerManager is @MainActor; send (small encode + network) hops on.
        Task { @MainActor in
            self.multipeerManager.send(payload)
        }

        // --- UI: publish raw normalized points for the live overlay ---
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.localKeypoints = uiPoints
            self.processedFrameCount &+= 1
        }
    }
}
