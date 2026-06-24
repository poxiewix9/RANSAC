//
//  CameraManager.swift
//  SpatialMap
//
//  Owns the AVCaptureSession and pumps frames through a video data output.
//
//  WHY AVCaptureVideoDataOutput (and not just a preview layer)?
//  We need access to the raw pixel buffers so that, in Phase 2, we can run the
//  Vision framework on them to extract sparse features. A preview-only setup
//  would show the camera but give us no buffers to analyze. So we configure a
//  `AVCaptureVideoDataOutput` whose delegate fires once per frame — that
//  delegate callback is the exact integration seam for the Vision pipeline.
//
//  CONCURRENCY MODEL
//  -----------------
//  This class is intentionally NOT @MainActor. The capture delegate callback
//  arrives on a background queue many times per second, and forcing that onto
//  the main actor would either be illegal (isolation errors) or kill
//  performance. Instead:
//    • Session config + start/stop run on a private serial `sessionQueue`.
//    • Frame callbacks run on a private serial `videoQueue`.
//    • Only the small `@Published` UI state is hopped to the main thread.
//  We mark the type `@unchecked Sendable` because we manually guarantee that
//  all shared mutable state is confined to the appropriate queue.
//

import Foundation
import AVFoundation
import CoreVideo
import Combine

final class CameraManager: NSObject, ObservableObject, @unchecked Sendable {

    // MARK: Observable state for the UI (always mutated on the main thread)

    /// Whether the capture session is currently running.
    @Published private(set) var isRunning = false

    /// Camera authorization status, so the UI can prompt / explain.
    @Published private(set) var authorizationStatus: AVAuthorizationStatus =
        AVCaptureDevice.authorizationStatus(for: .video)

    /// Human-readable error for the UI if setup fails.
    @Published private(set) var setupError: String?

    /// Rolling count of frames delivered — handy sanity check during bring-up.
    @Published private(set) var frameCount: Int = 0

    /// Most recent frame's pixel dimensions (set once the format is known).
    @Published private(set) var frameSize: CGSize = .zero

    // MARK: Capture plumbing

    /// The session is exposed (read-only) so the SwiftUI preview layer can bind
    /// to it. Everything that MUTATES it goes through `sessionQueue`.
    let session = AVCaptureSession()

    /// Serial queue for all session configuration + start/stop.
    private let sessionQueue = DispatchQueue(label: "com.ransac.SpatialMap.session")

    /// Serial queue on which per-frame sample buffers are delivered.
    private let videoQueue = DispatchQueue(label: "com.ransac.SpatialMap.video",
                                           qos: .userInitiated)

    private let videoOutput = AVCaptureVideoDataOutput()

    /// Phase 2 hook: anything assigned here is invoked once per frame with the
    /// raw pixel buffer, OFF the main thread. The VisionPipeline will plug in
    /// here to extract features without CameraManager knowing anything about
    /// Vision — keeping the modules decoupled. Confined to `videoQueue`.
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?

    /// Guarded by `sessionQueue`.
    private var isConfigured = false

    // MARK: Permissions

    /// Request camera access if needed, then configure + start the session.
    func startSession() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            publish { $0.authorizationStatus = .authorized }
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                self.publish {
                    $0.authorizationStatus = granted ? .authorized : .denied
                }
                if granted { self.configureAndStart() }
            }
        case .denied, .restricted:
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            publish {
                $0.authorizationStatus = status
                $0.setupError = "Camera access denied. Enable it in Settings → SpatialMap."
            }
        @unknown default:
            break
        }
    }

    /// Stop delivering frames and tear down the running session.
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning { self.session.stopRunning() }
            self.publish { $0.isRunning = false }
        }
    }

    // MARK: Configuration

    private func configureAndStart() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.isConfigured {
                self.configureSession()
            }
            guard self.isConfigured else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            let running = self.session.isRunning
            self.publish { $0.isRunning = running }
        }
    }

    /// One-time session wiring. Runs on `sessionQueue`.
    private func configureSession() {
        session.beginConfiguration()
        // 1080p is a good balance: plenty of detail for feature detection
        // without overloading the Vision pass. Tune later if needed.
        if session.canSetSessionPreset(.hd1920x1080) {
            session.sessionPreset = .hd1920x1080
        } else {
            session.sessionPreset = .high
        }

        // --- Camera input (rear wide-angle) ---
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                 for: .video,
                                                 position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            failConfiguration("Unable to access the rear camera.")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        // --- Video data output (gives us raw buffers) ---
        // 32BGRA is the most convenient pixel format for both display and for
        // handing to Vision/CoreImage later.
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // If Vision can't keep up, drop frames rather than queueing latency.
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        guard session.canAddOutput(videoOutput) else {
            failConfiguration("Unable to add video output.")
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)

        // Lock the connection to portrait so normalized coordinates have a
        // stable orientation for the geometry math later.
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90 // portrait
                }
            } else if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }

        session.commitConfiguration()
        isConfigured = true
    }

    /// Called on `sessionQueue` during configuration failures.
    private func failConfiguration(_ message: String) {
        isConfigured = false // safe: we are on sessionQueue here
        publish { $0.setupError = message }
    }

    // MARK: Main-thread publish helper

    /// Applies UI-state mutations on the main thread. Keeps every `@Published`
    /// write in one place so we never accidentally mutate from a queue.
    private func publish(_ mutate: @escaping (CameraManager) -> Void) {
        if Thread.isMainThread {
            mutate(self)
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                mutate(self)
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
//
// THIS is the Phase 2 integration point. Every camera frame lands here on
// `videoQueue`. Right now we just count frames and forward the pixel buffer to
// the optional `onFrame` hook. In Phase 2 the VisionPipeline subscribes to
// that hook, runs a VNImageRequestHandler on the buffer, extracts sparse
// feature points, packs them into a FeaturePayload, and hands it to
// MultipeerManager.send(_:).

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        // Forward to the Phase 2 consumer (Vision) without copying the buffer.
        onFrame?(pixelBuffer, pts)

        // Lightweight bookkeeping for the UI. Read dimensions cheaply.
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        publish {
            $0.frameCount &+= 1
            if $0.frameSize == .zero {
                $0.frameSize = CGSize(width: width, height: height)
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Frames are dropped when downstream work is too slow. Safe to ignore
        // in Phase 1; useful to log once the Vision pass exists.
    }
}
