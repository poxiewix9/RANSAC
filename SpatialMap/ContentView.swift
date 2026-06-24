//
//  ContentView.swift
//  SpatialMap
//
//  Phase 1 screen: live camera feed + a connection-status HUD overlay.
//
//  Layout is a ZStack:
//    • bottom layer  — the full-bleed camera preview
//    • top layer     — translucent HUD showing connection + link stats and a
//                      "Send Test Packet" button to validate the data link.
//
//  The two managers are pulled from the environment (created once in
//  SpatialMapApp), so this view never owns their lifecycle.
//

import SwiftUI
import UIKit
import AVFoundation

struct ContentView: View {
    @EnvironmentObject private var camera: CameraManager
    @EnvironmentObject private var multipeer: MultipeerManager

    var body: some View {
        ZStack {
            // --- Camera layer (or a fallback when not authorized) ---
            if camera.authorizationStatus == .authorized {
                CameraPreviewView(session: camera.session)
                    .ignoresSafeArea()
            } else {
                permissionPlaceholder
            }

            // --- HUD overlay ---
            VStack {
                statusBadge
                Spacer()
                linkStatsPanel
            }
            .padding()
        }
        // Kick off camera + networking when the screen appears; tear down when
        // it leaves so we release the camera and stop advertising politely.
        .onAppear {
            camera.startSession()
            multipeer.start()
        }
        .onDisappear {
            camera.stopSession()
            multipeer.stop()
        }
    }

    // MARK: Connection status badge (top)

    private var statusBadge: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 12, height: 12)
                .shadow(color: statusColor.opacity(0.8), radius: 4)
            Text(multipeer.state.rawValue)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Maps the coarse connection state to a traffic-light color.
    private var statusColor: Color {
        switch multipeer.state {
        case .connected:    return .green
        case .connecting:   return .yellow
        case .searching:    return .orange
        case .disconnected: return .red
        }
    }

    // MARK: Link stats + test controls (bottom)

    private var linkStatsPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if multipeer.connectedPeers.isEmpty {
                Label("No peers yet — open the app on a second device",
                      systemImage: "antenna.radiowaves.left.and.right")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.85))
            } else {
                Label("Peers: \(multipeer.connectedPeers.joined(separator: ", "))",
                      systemImage: "person.2.fill")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.white)
            }

            HStack(spacing: 16) {
                stat("Frames", "\(camera.frameCount)")
                stat("Sent", "\(multipeer.sentPayloadCount)")
                stat("Recv", "\(multipeer.receivedPayloadCount)")
            }

            if let last = multipeer.lastReceivedPayload {
                Text("Last packet: \(last.points.count) pts from \(last.senderName) (frame #\(last.frameID))")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.8))
                    .lineLimit(1)
            }

            // Phase 1 validation: send a synthetic FeaturePayload to peers.
            Button {
                multipeer.sendDummyPayload()
            } label: {
                Label("Send Test Packet", systemImage: "paperplane.fill")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(multipeer.connectedPeers.isEmpty)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.headline.monospacedDigit())
                .foregroundStyle(.white)
            Text(title)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.7))
        }
    }

    // MARK: Permission fallback

    private var permissionPlaceholder: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.metering.unknown")
                .font(.system(size: 48))
                .foregroundStyle(.white.opacity(0.6))
            Text(camera.setupError ?? "Camera access is required to map the scene.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.85))
                .padding(.horizontal, 40)
            if let url = URL(string: UIApplication.openSettingsURLString) {
                Link("Open Settings", destination: url)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
        .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
        .environmentObject(CameraManager())
        .environmentObject(MultipeerManager())
}
