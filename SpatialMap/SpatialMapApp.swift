//
//  SpatialMapApp.swift
//  SpatialMap
//
//  Decentralized Cross-Device Spatial Mapping.
//
//  PHASE 1 — Foundation: Multipeer Data Link + Camera Setup.
//
//  This is the app entry point. We construct the long-lived managers
//  (`CameraManager` and `MultipeerManager`) once here and inject them into
//  the SwiftUI environment so every view shares the same instances for the
//  lifetime of the app. These objects own real system resources
//  (an AVCaptureSession and an MCSession), so we never want SwiftUI to
//  recreate them on a view redraw.
//

import SwiftUI

@main
struct SpatialMapApp: App {
    // `@StateObject` guarantees these are instantiated exactly once and kept
    // alive for the whole app lifecycle, regardless of view churn.
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var multipeerManager = MultipeerManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(cameraManager)
                .environmentObject(multipeerManager)
        }
    }
}
