//
//  PointCloudView.swift
//  SpatialMap
//
//  Phase 5 — bridges the math engine's 3D output to a live SceneKit render.
//
//  SwiftUI has no native 3D scene view, so we wrap SCNView in a
//  UIViewRepresentable. The scene is built ONCE in makeUIView (camera, lights,
//  a persistent container node). Per update we only repopulate the container's
//  children — never rebuild the whole graph — which keeps it smooth.
//
//  COORDINATE / SCALE NOTE
//  -----------------------
//  Epipolar triangulation recovers structure only up to an unknown global scale
//  (monocular scale ambiguity), and our translation t is a unit vector. So the
//  triangulated coordinates are unitless. We multiply by `scale` purely so the
//  cloud is comfortably visible around the origin; it has no physical meaning
//  until a metric reference (e.g. known baseline) is introduced.
//

import SwiftUI
import SceneKit
import simd

struct PointCloudView: UIViewRepresentable {

    /// The triangulated 3D points to render. Driven by ContentView.
    @Binding var points: [simd_float3]

    /// Visualization scale factor (see scale note above).
    var scale: Float = 5.0

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        let scene = SCNScene()
        scnView.scene = scene
        scnView.allowsCameraControl = true          // pinch/drag to orbit
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor(white: 0.04, alpha: 1.0)
        scnView.antialiasingMode = .multisampling2X

        // --- Camera, pulled back along +Z so the origin is in view ---
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 5)
        scene.rootNode.addChildNode(cameraNode)

        // --- A faint origin marker for spatial reference ---
        let origin = SCNNode(geometry: SCNSphere(radius: 0.04))
        origin.geometry?.firstMaterial?.diffuse.contents = UIColor.darkGray
        origin.name = "origin"
        scene.rootNode.addChildNode(origin)

        // --- Persistent container we repopulate each update ---
        let container = SCNNode()
        container.name = "pointsRoot"
        scene.rootNode.addChildNode(container)
        context.coordinator.pointsRoot = container

        return scnView
    }

    func updateUIView(_ scnView: SCNView, context: Context) {
        guard let root = context.coordinator.pointsRoot else { return }

        // Clear previous frame's points (childNodes is a snapshot copy).
        root.childNodes.forEach { $0.removeFromParentNode() }

        // Share ONE glowing geometry across all point nodes — far cheaper than
        // building a unique geometry per sphere.
        let sphere = SCNSphere(radius: 0.03)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.cyan
        material.emission.contents = UIColor.cyan   // makes the dots "glow"
        material.lightingModel = .constant
        sphere.firstMaterial = material

        for p in points {
            let node = SCNNode(geometry: sphere)
            node.position = SCNVector3(p.x * scale, p.y * scale, p.z * scale)
            root.addChildNode(node)
        }
    }

    /// Holds the persistent node we mutate each update so we don't rebuild the
    /// whole scene graph every frame.
    final class Coordinator {
        var pointsRoot: SCNNode?
    }
}
