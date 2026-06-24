//
//  MultipeerManager.swift
//  SpatialMap
//
//  Owns peer discovery, session lifecycle, and (de)serialization of the
//  sparse feature payloads. This is the entire networking surface of the app.
//
//  DESIGN
//  ------
//  Each device simultaneously:
//    • ADVERTISES itself  (MCNearbyServiceAdvertiser) so others can find it.
//    • BROWSES for peers  (MCNearbyServiceBrowser)    so it can find others.
//  Both use the SAME serviceType ("cv-spatial-map"). When two symmetric peers
//  discover each other, both will try to invite. To avoid the classic
//  "double invite / connection thrash", we break the tie deterministically:
//  only the peer whose displayName is lexicographically smaller sends the
//  invite; the other simply waits and accepts. This makes auto-connect stable.
//
//  THREADING
//  ---------
//  MultipeerConnectivity delegate callbacks arrive on an internal queue. We
//  hop to the main actor before mutating any @Published state so SwiftUI stays
//  happy. Outbound sends happen off the main thread (the camera/vision queue
//  in later phases) — MCSession.send is itself thread-safe.
//

import Foundation
import MultipeerConnectivity
import os

/// High-level connection state surfaced to the UI.
enum ConnectionState: String {
    case searching = "Searching…"
    case connecting = "Connecting…"
    case connected = "Connected"
    case disconnected = "Disconnected"
}

@MainActor
final class MultipeerManager: NSObject, ObservableObject {

    // MARK: Public, observable state (drives the SwiftUI overlay)

    /// Coarse status string for the overlay badge.
    @Published private(set) var state: ConnectionState = .disconnected

    /// Display names of currently connected peers.
    @Published private(set) var connectedPeers: [String] = []

    /// Last payload received from a peer. In Phase 1 we just show its stats;
    /// in Phase 3 the GeometrySolver will consume this.
    @Published private(set) var lastReceivedPayload: FeaturePayload?

    /// Rolling counters useful for verifying the link during bring-up.
    @Published private(set) var sentPayloadCount: Int = 0
    @Published private(set) var receivedPayloadCount: Int = 0

    // MARK: Configuration

    /// MUST match the NSBonjourServices entries in Info.plist. Apple rules:
    /// 1–15 chars, lowercase ASCII letters/numbers/hyphens, no leading/trailing
    /// or consecutive hyphens.
    private static let serviceType = "cv-spatial-map"

    // MARK: Private MultipeerConnectivity plumbing

    /// Identifies THIS device on the mesh. Uses the device name for readability.
    private let myPeerID: MCPeerID

    /// The actual data pipe between connected peers.
    private let session: MCSession

    /// Tells nearby devices "I exist and offer this service."
    private let advertiser: MCNearbyServiceAdvertiser

    /// Looks for nearby devices offering the service.
    private let browser: MCNearbyServiceBrowser

    private let log = Logger(subsystem: "com.ransac.SpatialMap", category: "Multipeer")

    // MARK: Init

    override init() {
        // Use the device name so two phones are easy to tell apart in the UI.
        let peerID = MCPeerID(displayName: Self.makePeerName())
        self.myPeerID = peerID

        // `.required` enforces TLS-style encryption on the peer link. Good
        // hygiene even on a LAN; the perf cost is negligible for sparse data.
        self.session = MCSession(
            peer: peerID,
            securityIdentity: nil,
            encryptionPreference: .required
        )

        self.advertiser = MCNearbyServiceAdvertiser(
            peer: peerID,
            discoveryInfo: nil,
            serviceType: Self.serviceType
        )

        self.browser = MCNearbyServiceBrowser(
            peer: peerID,
            serviceType: Self.serviceType
        )

        super.init()

        // Wire up delegates. All three report back to us.
        session.delegate = self
        advertiser.delegate = self
        browser.delegate = self
    }

    deinit {
        // `stop()` is @MainActor; deinit may run off-main, so detach the work.
        advertiser.stopAdvertisingPeer()
        browser.stopBrowsingForPeers()
        session.disconnect()
    }

    // MARK: Lifecycle control

    /// Begin advertising + browsing. Call once the relevant view appears.
    func start() {
        log.info("Starting advertiser + browser as \(self.myPeerID.displayName, privacy: .public)")
        state = .searching
        advertiser.startAdvertisingPeer()
        browser.startBrowsingForPeers()
    }

    /// Tear everything down (e.g. when the view disappears or app backgrounds).
    func stop() {
        log.info("Stopping advertiser + browser")
        advertiser.stopAdvertisingPeer()
        browser.stopBrowsingForPeers()
        session.disconnect()
        connectedPeers = []
        state = .disconnected
    }

    // MARK: Sending

    /// Serialize and broadcast a feature payload to every connected peer.
    ///
    /// Uses `.reliable` for Phase 1 correctness while we validate the link.
    /// In Phase 3, high-rate feature streams will switch to `.unreliable`
    /// (datagram) so a single dropped frame never stalls the pipeline.
    func send(_ payload: FeaturePayload) {
        guard !session.connectedPeers.isEmpty else { return }
        do {
            let data = try payload.encoded()
            try session.send(data, toPeers: session.connectedPeers, with: .reliable)
            sentPayloadCount += 1
        } catch {
            log.error("Failed to send payload: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Phase 1 convenience: fire a synthetic payload to prove the pipe works.
    func sendDummyPayload() {
        let payload = FeaturePayload.dummy(
            frameID: UInt64(sentPayloadCount),
            senderName: myPeerID.displayName
        )
        send(payload)
    }

    // MARK: Helpers

    /// Builds a unique, human-readable peer name. We append a short random
    /// suffix so two devices with the same name still get distinct identities.
    private static func makePeerName() -> String {
        #if canImport(UIKit)
        let base = UIDevice.current.name
        #else
        let base = Host.current().localizedName ?? "Device"
        #endif
        // MCPeerID display names are capped at 63 UTF-8 bytes.
        let trimmed = String(base.prefix(48))
        let suffix = String(UUID().uuidString.prefix(4))
        return "\(trimmed)-\(suffix)"
    }

    /// Refresh the published peer list + coarse state from the live session.
    private func refreshConnectionState() {
        connectedPeers = session.connectedPeers.map(\.displayName)
        if !connectedPeers.isEmpty {
            state = .connected
        } else {
            // Still actively looking, so report "searching" rather than dead.
            state = .searching
        }
    }
}

#if canImport(UIKit)
import UIKit
#endif

// MARK: - MCSessionDelegate (the data pipe)

extension MultipeerManager: MCSessionDelegate {

    nonisolated func session(_ session: MCSession,
                             peer peerID: MCPeerID,
                             didChange state: MCSessionState) {
        // Hop to main before touching @Published properties.
        Task { @MainActor in
            switch state {
            case .connecting:
                self.state = .connecting
            case .connected:
                self.log.info("Connected to \(peerID.displayName, privacy: .public)")
                self.refreshConnectionState()
            case .notConnected:
                self.log.info("Lost \(peerID.displayName, privacy: .public)")
                self.refreshConnectionState()
            @unknown default:
                break
            }
        }
    }

    nonisolated func session(_ session: MCSession,
                             didReceive data: Data,
                             fromPeer peerID: MCPeerID) {
        // Decode OFF the main thread, then publish the result ON main.
        // (Decoding is cheap here, but keeping the pattern correct now means
        // the Phase 3 solver can do heavy work on this same background hop.)
        do {
            let payload = try FeaturePayload.decoded(from: data)
            Task { @MainActor in
                self.lastReceivedPayload = payload
                self.receivedPayloadCount += 1
            }
        } catch {
            Task { @MainActor in
                self.log.error("Decode failed from \(peerID.displayName, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    // Unused transports for Phase 1 — required by the protocol.
    nonisolated func session(_ session: MCSession,
                             didReceive stream: InputStream,
                             withName streamName: String,
                             fromPeer peerID: MCPeerID) {}

    nonisolated func session(_ session: MCSession,
                             didStartReceivingResourceWithName resourceName: String,
                             fromPeer peerID: MCPeerID,
                             with progress: Progress) {}

    nonisolated func session(_ session: MCSession,
                             didFinishReceivingResourceWithName resourceName: String,
                             fromPeer peerID: MCPeerID,
                             at localURL: URL?,
                             withError error: Error?) {}
}

// MARK: - MCNearbyServiceAdvertiserDelegate (we got invited)

extension MultipeerManager: MCNearbyServiceAdvertiserDelegate {

    nonisolated func advertiser(_ advertiser: MCNearbyServiceAdvertiser,
                                didReceiveInvitationFromPeer peerID: MCPeerID,
                                withContext context: Data?,
                                invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        // Auto-accept any invitation into our single shared session.
        // The tie-break logic on the browser side prevents duplicate links.
        Task { @MainActor in
            self.log.info("Accepting invite from \(peerID.displayName, privacy: .public)")
        }
        invitationHandler(true, session)
    }

    nonisolated func advertiser(_ advertiser: MCNearbyServiceAdvertiser,
                                didNotStartAdvertisingPeer error: Error) {
        Task { @MainActor in
            self.log.error("Advertiser failed: \(error.localizedDescription, privacy: .public)")
        }
    }
}

// MARK: - MCNearbyServiceBrowserDelegate (we found / lost peers)

extension MultipeerManager: MCNearbyServiceBrowserDelegate {

    nonisolated func browser(_ browser: MCNearbyServiceBrowser,
                             foundPeer peerID: MCPeerID,
                             withDiscoveryInfo info: [String: String]?) {
        // Deterministic tie-break: only the "smaller" name initiates the invite
        // so both peers don't invite each other and create a connection race.
        let shouldInvite = myPeerID.displayName < peerID.displayName
        Task { @MainActor in
            if shouldInvite {
                self.log.info("Inviting \(peerID.displayName, privacy: .public)")
                browser.invitePeer(peerID, to: self.session, withContext: nil, timeout: 15)
            } else {
                self.log.info("Found \(peerID.displayName, privacy: .public); waiting for their invite")
            }
            if self.state == .searching { self.state = .connecting }
        }
    }

    nonisolated func browser(_ browser: MCNearbyServiceBrowser,
                             lostPeer peerID: MCPeerID) {
        Task { @MainActor in
            self.log.info("Lost sight of \(peerID.displayName, privacy: .public)")
            self.refreshConnectionState()
        }
    }

    nonisolated func browser(_ browser: MCNearbyServiceBrowser,
                             didNotStartBrowsingForPeers error: Error) {
        Task { @MainActor in
            self.log.error("Browser failed: \(error.localizedDescription, privacy: .public)")
        }
    }
}
