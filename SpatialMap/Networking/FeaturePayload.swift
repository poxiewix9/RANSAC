//
//  FeaturePayload.swift
//  SpatialMap
//
//  The lightweight, wire-format data contract exchanged between peers.
//
//  ARCHITECTURE CONSTRAINT (critical):
//  We NEVER send CMSampleBuffers, CVPixelBuffers, JPEGs, or any raw imagery
//  across the network. We only send sparse mathematical descriptions of the
//  scene: 2D feature coordinates and (eventually) their descriptor vectors.
//  This keeps each packet in the low-kilobyte range instead of megabytes,
//  which is what makes real-time peer-to-peer exchange feasible.
//

import Foundation
import simd

/// A single detected interest point in one camera frame.
///
/// Coordinates are stored in **normalized image space** (0...1, origin at the
/// top-left) so that the receiving device can interpret them independently of
/// the sender's pixel resolution. The `GeometrySolver` in Phase 3 will later
/// lift these into homogeneous camera coordinates using each device's
/// intrinsic matrix.
struct FeaturePoint: Codable, Hashable {
    /// Normalized x in [0, 1].
    var x: Float
    /// Normalized y in [0, 1].
    var y: Float

    /// Optional compact descriptor for this point (e.g. a feature-print slice).
    /// In Phase 1 this is empty/placeholder; Phase 2 fills it from the Vision
    /// framework so the receiver can perform robust feature matching.
    var descriptor: [Float]

    init(x: Float, y: Float, descriptor: [Float] = []) {
        self.x = x
        self.y = y
        self.descriptor = descriptor
    }

    /// Convenience bridge to `simd` for the math-heavy Phase 3 code.
    var simdPoint: SIMD2<Float> { SIMD2<Float>(x, y) }
}

/// One frame's worth of features, plus the metadata the solver needs to make
/// sense of them. This is the unit we serialize and stream over Multipeer.
struct FeaturePayload: Codable {
    /// Schema version so we can evolve the wire format without breaking peers.
    var version: Int

    /// Monotonic frame counter from the sender. Lets the receiver pair frames
    /// and reason about temporal ordering / drop stale packets.
    var frameID: UInt64

    /// Capture timestamp (seconds, sender clock). Useful for loose temporal
    /// alignment between two unsynchronized devices.
    var timestamp: TimeInterval

    /// The sender's display name (mirrors its MCPeerID). Handy for debugging
    /// and for labeling which camera a feature set came from.
    var senderName: String

    /// Source frame dimensions in pixels. Combined with normalized coords this
    /// lets the receiver reconstruct pixel positions and, with intrinsics,
    /// homogeneous rays for triangulation.
    var imageWidth: Int
    var imageHeight: Int

    /// The actual sparse features detected this frame.
    var points: [FeaturePoint]

    init(
        version: Int = FeaturePayload.currentVersion,
        frameID: UInt64,
        timestamp: TimeInterval = Date().timeIntervalSince1970,
        senderName: String,
        imageWidth: Int,
        imageHeight: Int,
        points: [FeaturePoint]
    ) {
        self.version = version
        self.frameID = frameID
        self.timestamp = timestamp
        self.senderName = senderName
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.points = points
    }

    static let currentVersion = 1
}

// MARK: - Serialization helpers

extension FeaturePayload {
    /// Encodes the payload to `Data` for `MCSession.send(_:toPeers:with:)`.
    ///
    /// We use `PropertyListEncoder` (binary) here because it is compact and
    /// fast for flat numeric structs. If we later need cross-platform interop
    /// we can swap in `JSONEncoder` or a protobuf without touching call sites.
    func encoded() throws -> Data {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        return try encoder.encode(self)
    }

    /// Decodes a payload received from a peer.
    static func decoded(from data: Data) throws -> FeaturePayload {
        try PropertyListDecoder().decode(FeaturePayload.self, from: data)
    }

    /// Phase 1 test helper: fabricate a payload full of random 2D points so we
    /// can verify the data link end-to-end before the Vision pipeline exists.
    static func dummy(frameID: UInt64, senderName: String, count: Int = 32) -> FeaturePayload {
        let points = (0..<count).map { _ in
            FeaturePoint(x: .random(in: 0...1), y: .random(in: 0...1))
        }
        return FeaturePayload(
            frameID: frameID,
            senderName: senderName,
            imageWidth: 1920,
            imageHeight: 1080,
            points: points
        )
    }
}
