//
//  CameraPreviewView.swift
//  SpatialMap
//
//  A thin SwiftUI wrapper around AVCaptureVideoPreviewLayer.
//
//  SwiftUI has no native camera preview, so we bridge UIKit via
//  UIViewRepresentable. We back the view with a UIView whose `layerClass` is
//  AVCaptureVideoPreviewLayer — this is the canonical, GPU-efficient way to
//  display a live AVCaptureSession with zero per-frame CPU work on our side.
//
//  NOTE: This layer is for DISPLAY only. The pixel buffers we actually analyze
//  come from CameraManager's AVCaptureVideoDataOutput, not from this layer.
//

import SwiftUI
import UIKit
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {

    /// The session to display. Provided by CameraManager.
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        // Fill the screen while preserving aspect ratio (crops edges).
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // Keep the session reference current if it ever changes.
        if uiView.videoPreviewLayer.session !== session {
            uiView.videoPreviewLayer.session = session
        }
    }

    /// UIView whose root layer IS the preview layer — most efficient setup.
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            // Safe: layerClass guarantees this exact type.
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
