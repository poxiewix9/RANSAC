//
//  GeometrySolver.swift
//  SpatialMap
//
//  Phase 3 — The Math Engine: epipolar geometry from two sets of 2D features.
//
//  GIVEN matched 2D points seen by two cameras, this recovers:
//    1. the Essential matrix E (the epipolar relationship between the views),
//    2. the relative pose (rotation R, translation t) of camera 2 vs camera 1,
//    3. the 3D position of each matched point via triangulation.
//
//  PIPELINE
//  --------
//    matchFeatures            naive spatial NN (PLACEHOLDER — see note below)
//        ↓ [(x1, x2)] pairs
//    estimateEssentialMatrix  RANSAC + 8-point algorithm + SVD (LAPACK)
//        ↓ E (3x3, rank 2)
//    extractPose              SVD(E) + the W trick → (R, t)
//        ↓
//    triangulate              Direct Linear Transform (DLT) per point → 3D
//
//  COORDINATE / CALIBRATION NOTE
//  -----------------------------
//  We operate directly on the Vision-normalized image coordinates (0...1,
//  bottom-left origin) WITHOUT applying camera intrinsics K. Strictly, the
//  "Essential" matrix requires calibrated rays (x = K⁻¹·pixel); what we compute
//  here is really a Fundamental-style estimate treated as E. That is fine for
//  Phase 3's goal — proving the data flow and producing 3D points — and a real
//  K can be slotted in later (Phase 4+) by pre-multiplying points by K⁻¹.
//
//  NUMERICAL NOTE
//  --------------
//  The classic 8-point algorithm benefits from Hartley isotropic normalization.
//  Because our coordinates already live in a tidy [0,1] range (unlike raw pixels
//  in the thousands), conditioning is acceptable without it. Adding Hartley
//  normalization is a clean future improvement.
//

import Foundation
import simd
import Accelerate

struct GeometrySolver {

    // MARK: Tunable parameters

    /// Number of RANSAC hypotheses to try.
    var ransacIterations: Int = 500

    /// Sampson-distance threshold (in normalized-coord units²) below which a
    /// match is considered an inlier. Tune as matching quality improves.
    var sampsonThreshold: Float = 1e-3

    /// Lowe's ratio-test threshold. A match is kept only if the best descriptor
    /// SSD is below this fraction of the second-best — rejecting ambiguous
    /// matches (e.g. repeating textures) that would otherwise pollute RANSAC.
    var loweRatio: Float = 0.75

    // MARK: - 1. Matching (descriptor SSD + Lowe's ratio test)

    /// Matches features by appearance using their 121-float patch descriptors.
    ///
    /// For each local feature we find the two closest remote descriptors by Sum
    /// of Squared Differences (SSD), then apply Lowe's ratio test: keep the
    /// match only if `bestSSD < loweRatio * secondBestSSD`. This is robust to
    /// viewpoint/lighting changes (the patches are illumination-normalized) and
    /// discards ambiguous corners with no clear single best match.
    func matchFeatures(local: [FeaturePoint],
                       remote: [FeaturePoint]) -> [(simd_float2, simd_float2)] {
        guard !local.isEmpty, !remote.isEmpty else { return [] }

        var matches: [(simd_float2, simd_float2)] = []
        matches.reserveCapacity(local.count)

        for l in local {
            guard !l.descriptor.isEmpty else { continue }

            var bestSSD = Float.greatestFiniteMagnitude
            var secondSSD = Float.greatestFiniteMagnitude
            var bestRemote: FeaturePoint?

            for r in remote {
                guard r.descriptor.count == l.descriptor.count else { continue }
                let d = ssd(l.descriptor, r.descriptor)
                if d < bestSSD {
                    secondSSD = bestSSD
                    bestSSD = d
                    bestRemote = r
                } else if d < secondSSD {
                    secondSSD = d
                }
            }

            // Lowe's ratio test.
            if let r = bestRemote, bestSSD < loweRatio * secondSSD {
                matches.append((l.simdPoint, r.simdPoint))
            }
        }
        return matches
    }

    /// Sum of squared differences between two equal-length descriptors.
    private func ssd(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let d = a[i] - b[i]
            sum += d * d
        }
        return sum
    }

    // MARK: - Camera intrinsics

    /// Builds a placeholder pinhole intrinsic matrix K for a given frame size.
    ///
    /// We approximate a typical iPhone rear camera: focal length ≈ 0.8·width in
    /// pixels and the principal point at the image center. Replace this with
    /// real values from `AVCameraCalibrationData` when available for accuracy.
    ///
    ///   K = [ fx  0  cx ]
    ///       [  0 fy  cy ]
    ///       [  0  0   1 ]
    func intrinsics(width: Int, height: Int) -> simd_float3x3 {
        let w = Float(width)
        let h = Float(height)
        let f = w * 0.8
        return simd_float3x3(rows: [
            SIMD3<Float>(f, 0, w / 2),
            SIMD3<Float>(0, f, h / 2),
            SIMD3<Float>(0, 0, 1)
        ])
    }

    /// Un-projects a normalized (0...1) image point into calibrated camera
    /// coordinates: x̂ = K⁻¹ · [u·W, v·H, 1]ᵀ. After this, the 8-point algorithm
    /// recovers a true Essential matrix rather than a Fundamental one.
    private func unproject(_ p: simd_float2,
                           kInverse: simd_float3x3,
                           width: Int,
                           height: Int) -> simd_float2 {
        let pixel = SIMD3<Float>(p.x * Float(width), p.y * Float(height), 1)
        let c = kInverse * pixel
        return SIMD2<Float>(c.x / c.z, c.y / c.z)
    }

    // MARK: - 2. Essential Matrix via RANSAC + 8-point algorithm

    /// Robustly estimates E from noisy matches. Each RANSAC iteration solves the
    /// 8-point algorithm on a random minimal sample, then scores it by inlier
    /// count using the Sampson distance. The best hypothesis is refit on all of
    /// its inliers.
    func estimateEssentialMatrix(matches: [(simd_float2, simd_float2)]) -> simd_float3x3? {
        let n = matches.count
        guard n >= 8 else { return nil }

        var bestE: simd_float3x3?
        var bestInlierCount = -1

        for _ in 0..<ransacIterations {
            let sample = randomSample(count: 8, from: n)
            guard let candidate = eightPoint(matches: sample.map { matches[$0] }) else {
                continue
            }
            let inliers = countInliers(candidate, matches: matches)
            if inliers > bestInlierCount {
                bestInlierCount = inliers
                bestE = candidate
            }
        }

        guard let coarseE = bestE else { return nil }

        // Refit on the full inlier set for a tighter estimate.
        let inlierMatches = matches.filter {
            sampsonDistance(coarseE, x1: $0.0, x2: $0.1) < sampsonThreshold
        }
        if inlierMatches.count >= 8, let refined = eightPoint(matches: inlierMatches) {
            return refined
        }
        return coarseE
    }

    /// The normalized 8-point algorithm: builds the constraint matrix A from the
    /// epipolar equation x2ᵀ·E·x1 = 0, finds E as the null-space of A (smallest
    /// right singular vector), then forces E to rank 2.
    private func eightPoint(matches: [(simd_float2, simd_float2)]) -> simd_float3x3? {
        let n = matches.count
        guard n >= 8 else { return nil }

        // Build A (n x 9), row-major. Each row encodes x2ᵀ E x1 = 0 for one pair.
        var a = [Float](repeating: 0, count: n * 9)
        for (i, m) in matches.enumerated() {
            let (x1, x2) = m
            let u1 = x1.x, v1 = x1.y
            let u2 = x2.x, v2 = x2.y
            // Column order matches a row-major flatten of E = [e11..e33].
            let row: [Float] = [u2 * u1, u2 * v1, u2,
                                v2 * u1, v2 * v1, v2,
                                u1,      v1,      1]
            for c in 0..<9 { a[i * 9 + c] = row[c] }
        }

        // Solve A·f = 0  →  f is the right singular vector of the SMALLEST
        // singular value, i.e. the LAST row of Vᵀ.
        guard let svd = svd(a, rows: n, cols: 9) else { return nil }
        let f = Array(svd.vt[(8 * 9)..<(9 * 9)])   // last (9th) row of Vᵀ
        let rawE = matrix(rowMajor: f)

        // Enforce the rank-2 constraint that every valid E/F must satisfy.
        return enforceRank2(rawE)
    }

    /// Projects E onto the closest rank-2 matrix by zeroing its smallest
    /// singular value: E = U · diag(σ1, σ2, 0) · Vᵀ.
    private func enforceRank2(_ E: simd_float3x3) -> simd_float3x3 {
        guard let svd = svd(rowMajor(E), rows: 3, cols: 3) else { return E }
        let U = matrix(rowMajor: svd.u)
        let Vt = matrix(rowMajor: svd.vt)
        let D = simd_float3x3(diagonal: SIMD3<Float>(svd.s[0], svd.s[1], 0))
        return U * D * Vt
    }

    // MARK: - 3. Pose extraction (R, t) from E

    /// Decomposes E into the physically correct relative pose using the
    /// CHEIRALITY CHECK. SVD(E) yields two candidate rotations (R₁ = U·W·Vᵀ,
    /// R₂ = U·Wᵀ·Vᵀ) and two translations (±t, where t is the 3rd column of U),
    /// giving four (R, t) pairs. Only one places the scene IN FRONT of both
    /// cameras, so we triangulate the matches under each candidate and keep the
    /// pair that maximizes the count of points with positive depth (Z > 0) in
    /// BOTH camera frames.
    ///
    /// `matches` must be in calibrated (un-projected) coordinates — the same
    /// space used to estimate E.
    func extractPose(from E: simd_float3x3,
                     matches: [(simd_float2, simd_float2)]) -> (R: simd_float3x3, t: simd_float3) {
        guard let svd = svd(rowMajor(E), rows: 3, cols: 3) else {
            return (matrix_identity_float3x3, SIMD3<Float>(0, 0, 1))
        }
        var U = matrix(rowMajor: svd.u)
        var Vt = matrix(rowMajor: svd.vt)

        // Make U and V proper rotations so the candidate R's are too.
        if determinant(U) < 0 { U = scaled(U, -1) }
        if determinant(Vt) < 0 { Vt = scaled(Vt, -1) }

        let W = matrix(rowMajor: [0, -1, 0,
                                  1,  0, 0,
                                  0,  0, 1])

        var R1 = U * W * Vt
        var R2 = U * W.transpose * Vt
        if determinant(R1) < 0 { R1 = scaled(R1, -1) }
        if determinant(R2) < 0 { R2 = scaled(R2, -1) }

        let t = U.columns.2

        // The four physically distinct hypotheses.
        let candidates: [(R: simd_float3x3, t: simd_float3)] = [
            (R1,  t), (R1, -t),
            (R2,  t), (R2, -t)
        ]

        var best = candidates[0]
        var bestInFront = -1
        for candidate in candidates {
            let count = pointsInFront(matches: matches, R: candidate.R, t: candidate.t)
            if count > bestInFront {
                bestInFront = count
                best = candidate
            }
        }
        return best
    }

    /// Triangulates the matches under a candidate pose and counts how many 3D
    /// points lie in front of BOTH cameras (positive Z in each frame).
    ///   • Camera 1 frame: the point X itself (P1 = [I|0]) → check X.z > 0.
    ///   • Camera 2 frame: X' = R·X + t                    → check X'.z > 0.
    private func pointsInFront(matches: [(simd_float2, simd_float2)],
                               R: simd_float3x3,
                               t: simd_float3) -> Int {
        let points = triangulate(matches: matches, R: R, t: t)
        var count = 0
        for X in points {
            let xInCam2 = R * X + t
            if X.z > 0 && xInCam2.z > 0 { count += 1 }
        }
        return count
    }

    // MARK: - 4. Triangulation (DLT)

    /// Recovers 3D points via the Direct Linear Transform. Camera 1 is the world
    /// origin with projection P1 = [I | 0]; camera 2 is P2 = [R | t]. For each
    /// match we stack the 4 linear constraints and take the null space (smallest
    /// right singular vector) as the homogeneous 3D point.
    func triangulate(matches: [(simd_float2, simd_float2)],
                     R: simd_float3x3,
                     t: simd_float3) -> [simd_float3] {
        // Rows of P1 = [I | 0].
        let p1r0 = SIMD4<Float>(1, 0, 0, 0)
        let p1r1 = SIMD4<Float>(0, 1, 0, 0)
        let p1r2 = SIMD4<Float>(0, 0, 1, 0)

        // Rows of P2 = [R | t].
        let r0 = rowOf(R, 0), r1 = rowOf(R, 1), r2 = rowOf(R, 2)
        let p2r0 = SIMD4<Float>(r0.x, r0.y, r0.z, t.x)
        let p2r1 = SIMD4<Float>(r1.x, r1.y, r1.z, t.y)
        let p2r2 = SIMD4<Float>(r2.x, r2.y, r2.z, t.z)

        var points: [simd_float3] = []
        points.reserveCapacity(matches.count)

        for (x1, x2) in matches {
            // A·X = 0, A is 4x4: each image gives two rows.
            let a0 = x1.x * p1r2 - p1r0
            let a1 = x1.y * p1r2 - p1r1
            let a2 = x2.x * p2r2 - p2r0
            let a3 = x2.y * p2r2 - p2r1

            var a = [Float](repeating: 0, count: 16)
            store(row: a0, into: &a, at: 0)
            store(row: a1, into: &a, at: 1)
            store(row: a2, into: &a, at: 2)
            store(row: a3, into: &a, at: 3)

            guard let svd = svd(a, rows: 4, cols: 4) else { continue }
            // Homogeneous solution = last row of Vᵀ.
            let X = Array(svd.vt[(3 * 4)..<(4 * 4)])
            let w = X[3]
            guard abs(w) > 1e-7 else { continue }
            points.append(SIMD3<Float>(X[0] / w, X[1] / w, X[2] / w))
        }
        return points
    }

    // MARK: - 5. Main pipeline

    /// Chains the full geometry pipeline:
    ///   match (descriptors) → un-project with K → E (RANSAC/SVD) →
    ///   pose (cheirality) → triangulate.
    /// Returns the triangulated 3D points (empty if anything along the way
    /// fails, e.g. too few matches/inliers).
    func process(localPayload: FeaturePayload,
                 remotePayload: FeaturePayload) -> [simd_float3] {
        // 1. Appearance matching in raw normalized (0...1) image coordinates.
        let matches = matchFeatures(local: localPayload.points,
                                    remote: remotePayload.points)
        guard matches.count >= 8 else { return [] }

        // 2. Un-project both views into calibrated coordinates using each
        //    camera's own intrinsics, so we estimate a true Essential matrix.
        let kLocalInv = intrinsics(width: localPayload.imageWidth,
                                   height: localPayload.imageHeight).inverse
        let kRemoteInv = intrinsics(width: remotePayload.imageWidth,
                                    height: remotePayload.imageHeight).inverse
        let calibrated: [(simd_float2, simd_float2)] = matches.map { m in
            let a = unproject(m.0, kInverse: kLocalInv,
                              width: localPayload.imageWidth,
                              height: localPayload.imageHeight)
            let b = unproject(m.1, kInverse: kRemoteInv,
                              width: remotePayload.imageWidth,
                              height: remotePayload.imageHeight)
            return (a, b)
        }

        // 3. Robust Essential matrix.
        guard let E = estimateEssentialMatrix(matches: calibrated) else { return [] }

        // 4. Keep only the geometric inliers.
        let inliers = calibrated.filter {
            sampsonDistance(E, x1: $0.0, x2: $0.1) < sampsonThreshold
        }
        guard inliers.count >= 8 else { return [] }

        // 5. Disambiguate the pose via cheirality, then triangulate.
        let pose = extractPose(from: E, matches: inliers)
        return triangulate(matches: inliers, R: pose.R, t: pose.t)
    }

    // MARK: - Epipolar scoring

    /// Sampson distance: a first-order geometric approximation of the squared
    /// reprojection error of a match w.r.t. the epipolar constraint xᵀ₂·E·x₁.
    private func sampsonDistance(_ E: simd_float3x3,
                                 x1: simd_float2,
                                 x2: simd_float2) -> Float {
        let x1h = SIMD3<Float>(x1.x, x1.y, 1)
        let x2h = SIMD3<Float>(x2.x, x2.y, 1)
        let Ex1 = E * x1h            // epipolar line in image 2
        let Etx2 = E.transpose * x2h // epipolar line in image 1
        let numer = simd_dot(x2h, Ex1)        // x2ᵀ E x1
        let denom = Ex1.x * Ex1.x + Ex1.y * Ex1.y
                  + Etx2.x * Etx2.x + Etx2.y * Etx2.y
        guard denom > 0 else { return .greatestFiniteMagnitude }
        return (numer * numer) / denom
    }

    private func countInliers(_ E: simd_float3x3,
                              matches: [(simd_float2, simd_float2)]) -> Int {
        var count = 0
        for (x1, x2) in matches where sampsonDistance(E, x1: x1, x2: x2) < sampsonThreshold {
            count += 1
        }
        return count
    }

    // MARK: - RANSAC sampling

    /// Picks `count` distinct random indices in 0..<n.
    private func randomSample(count: Int, from n: Int) -> [Int] {
        if n <= count { return Array(0..<n) }
        var chosen = Set<Int>()
        chosen.reserveCapacity(count)
        while chosen.count < count {
            chosen.insert(Int.random(in: 0..<n))
        }
        return Array(chosen)
    }

    // MARK: - LAPACK SVD wrapper

    /// Full singular value decomposition A = U·Σ·Vᵀ of a row-major m×n matrix.
    ///
    /// Returns U (m×m, row-major), the singular values s (descending), and Vᵀ
    /// (n×n, row-major). Returns nil if LAPACK reports a failure.
    ///
    /// LAPACK is Fortran column-major, so we transpose our row-major input in,
    /// and transpose U/Vᵀ back out. We use `Int32` for all integer arguments:
    /// it matches both the legacy `__CLPK_integer` and the modern (non-ILP64)
    /// `__LAPACK_int`, so this compiles regardless of which Accelerate LAPACK
    /// interface is active.
    private func svd(_ rowMajor: [Float], rows m: Int, cols n: Int)
        -> (u: [Float], s: [Float], vt: [Float])? {
        guard m > 0, n > 0 else { return nil }

        var a = Self.rowToColMajor(rowMajor, rows: m, cols: n)  // column-major in

        var jobu: CChar = 0x41   // 'A' — return all m columns of U
        var jobvt: CChar = 0x41  // 'A' — return all n rows of Vᵀ
        var mm = Int32(m)
        var nn = Int32(n)
        var lda = Int32(m)
        var ldu = Int32(m)
        var ldvt = Int32(n)
        var info = Int32(0)

        let k = Swift.min(m, n)
        var s = [Float](repeating: 0, count: k)
        var u = [Float](repeating: 0, count: m * m)   // column-major m×m
        var vt = [Float](repeating: 0, count: n * n)  // column-major n×n

        // 1) Workspace size query (lwork = -1).
        var workQuery = Float(0)
        var lwork = Int32(-1)
        sgesvd_(&jobu, &jobvt, &mm, &nn, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &workQuery, &lwork, &info)
        guard info == 0 else { return nil }

        // 2) Allocate workspace and run the real decomposition.
        lwork = Int32(workQuery)
        var work = [Float](repeating: 0, count: Int(max(1, lwork)))
        sgesvd_(&jobu, &jobvt, &mm, &nn, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &info)
        guard info == 0 else { return nil }

        // Convert column-major outputs back to row-major.
        return (Self.colToRowMajor(u, rows: m, cols: m),
                s,
                Self.colToRowMajor(vt, rows: n, cols: n))
    }

    // MARK: - Layout helpers (row-major ⇄ column-major)

    private static func rowToColMajor(_ a: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                out[i + j * rows] = a[i * cols + j]
            }
        }
        return out
    }

    private static func colToRowMajor(_ a: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                out[i * cols + j] = a[i + j * rows]
            }
        }
        return out
    }

    // MARK: - simd helpers

    /// Builds a 3x3 matrix from 9 row-major values.
    private func matrix(rowMajor a: [Float]) -> simd_float3x3 {
        simd_float3x3(rows: [
            SIMD3<Float>(a[0], a[1], a[2]),
            SIMD3<Float>(a[3], a[4], a[5]),
            SIMD3<Float>(a[6], a[7], a[8])
        ])
    }

    /// Flattens a 3x3 matrix to 9 row-major values (simd stores column-major).
    private func rowMajor(_ m: simd_float3x3) -> [Float] {
        var a = [Float](repeating: 0, count: 9)
        for r in 0..<3 {
            for c in 0..<3 {
                a[r * 3 + c] = m[c][r]   // m[column][row]
            }
        }
        return a
    }

    /// Row `r` of a 3x3 matrix (simd subscript is [column][row]).
    private func rowOf(_ m: simd_float3x3, _ r: Int) -> simd_float3 {
        SIMD3<Float>(m[0][r], m[1][r], m[2][r])
    }

    private func scaled(_ m: simd_float3x3, _ s: Float) -> simd_float3x3 {
        simd_float3x3(columns: (m.columns.0 * s, m.columns.1 * s, m.columns.2 * s))
    }

    /// det via the scalar triple product of the columns.
    private func determinant(_ m: simd_float3x3) -> Float {
        simd_dot(m.columns.0, simd_cross(m.columns.1, m.columns.2))
    }

    private func store(row v: SIMD4<Float>, into a: inout [Float], at r: Int) {
        a[r * 4 + 0] = v.x
        a[r * 4 + 1] = v.y
        a[r * 4 + 2] = v.z
        a[r * 4 + 3] = v.w
    }
}
