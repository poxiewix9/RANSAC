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

    /// Hard cap on points streamed per frame. Keeps each payload in the low-KB
    /// range regardless of how busy the scene is.
    private let maxPointsPerFrame = 200

    /// Vision processes a downscaled copy at this longest-edge size. Smaller =
    /// faster + fewer, cleaner contours. Detail is plenty for sparse features.
    private let maxImageDimension = 512

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

        // --- 4. DISPATCH -----------------------------------------------------
        frameID &+= 1
        dispatch(points: points, frameID: frameID, width: width, height: height)
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

    private func dispatch(points: [CGPoint], frameID: UInt64, width: Int, height: Int) {
        // --- Network: build payload and send via the (main-actor) manager ---
        // Points are mapped to FeaturePoint with an empty descriptor for now;
        // Phase 3's matcher will fill descriptors in.
        let featurePoints = points.map {
            FeaturePoint(x: Float($0.x), y: Float($0.y))
        }
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
            self.localKeypoints = points
            self.processedFrameCount &+= 1
        }
    }
}
