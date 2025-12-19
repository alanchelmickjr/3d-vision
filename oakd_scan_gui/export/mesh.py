"""Mesh and point cloud export utilities."""

from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


class MeshExporter:
    """Export point clouds to various mesh formats."""

    SUPPORTED_FORMATS = ["ply", "stl", "obj", "pcd", "xyz"]

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./scans")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        format_type: str = "ply",
        filename: str = None,
        create_mesh: bool = False,
    ) -> Path:
        """
        Export point cloud to file.

        Args:
            points: 3D points (N, 3)
            colors: RGB colors (N, 3), values in 0-1 range
            format_type: Output format (ply, stl, obj, pcd, xyz)
            filename: Optional custom filename
            create_mesh: Whether to create mesh from points

        Returns:
            Path to exported file
        """
        if not OPEN3D_AVAILABLE:
            return self._export_simple(points, colors, format_type, filename)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_{timestamp}"

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            # Ensure colors are in 0-1 range
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Clean up point cloud
        pcd = self._cleanup_pointcloud(pcd)

        if create_mesh or format_type in ["stl", "obj"]:
            # Create mesh from points
            mesh = self._create_mesh(pcd)
            return self._export_mesh(mesh, format_type, filename)
        else:
            return self._export_pointcloud(pcd, format_type, filename)

    def _cleanup_pointcloud(self, pcd) -> "o3d.geometry.PointCloud":
        """Remove outliers and noise from point cloud."""
        # Statistical outlier removal
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        # Voxel downsampling for efficiency (0.5mm voxel size)
        pcd_down = pcd_clean.voxel_down_sample(voxel_size=0.0005)

        return pcd_down

    def _create_mesh(self, pcd) -> "o3d.geometry.TriangleMesh":
        """Create mesh from point cloud using Poisson reconstruction."""
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )

        # Remove low-density vertices (artifacts)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.1)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        return mesh

    def _export_pointcloud(
        self,
        pcd: "o3d.geometry.PointCloud",
        format_type: str,
        filename: str,
    ) -> Path:
        """Export point cloud to file."""
        ext = format_type.lower()
        output_path = self.output_dir / f"{filename}.{ext}"

        if ext == "ply":
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=True)
        elif ext == "pcd":
            o3d.io.write_point_cloud(str(output_path), pcd)
        elif ext == "xyz":
            # Simple XYZ format
            points = np.asarray(pcd.points)
            np.savetxt(output_path, points, fmt="%.6f")
        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")

        print(f"Exported point cloud: {output_path}")
        return output_path

    def _export_mesh(
        self,
        mesh: "o3d.geometry.TriangleMesh",
        format_type: str,
        filename: str,
    ) -> Path:
        """Export mesh to file."""
        ext = format_type.lower()
        output_path = self.output_dir / f"{filename}.{ext}"

        if ext == "ply":
            o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=True)
        elif ext == "stl":
            o3d.io.write_triangle_mesh(str(output_path), mesh)
        elif ext == "obj":
            o3d.io.write_triangle_mesh(str(output_path), mesh)
        else:
            raise ValueError(f"Unsupported mesh format: {ext}")

        print(f"Exported mesh: {output_path}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Triangles: {len(mesh.triangles)}")
        return output_path

    def _export_simple(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        format_type: str,
        filename: str,
    ) -> Path:
        """Fallback export without Open3D."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_{timestamp}"

        ext = format_type.lower()
        output_path = self.output_dir / f"{filename}.{ext}"

        if ext in ["ply", "xyz"]:
            self._write_ply(output_path, points, colors)
        else:
            print(f"Warning: {ext} format requires Open3D, exporting as PLY")
            output_path = output_path.with_suffix(".ply")
            self._write_ply(output_path, points, colors)

        print(f"Exported: {output_path} ({len(points)} points)")
        return output_path

    def _write_ply(
        self,
        path: Path,
        points: np.ndarray,
        colors: np.ndarray = None,
    ):
        """Write PLY file without Open3D."""
        n_points = len(points)

        has_color = colors is not None and len(colors) == n_points

        with open(path, "w") as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            # Data
            for i in range(n_points):
                x, y, z = points[i]
                if has_color:
                    r, g, b = colors[i]
                    if r <= 1.0:  # Normalize if in 0-1 range
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


class ScanSession:
    """Manages a scanning session with multiple captures."""

    def __init__(self, output_dir: Path = None, session_name: str = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"

        self.output_dir = (
            Path(output_dir) if output_dir else Path("./scans")
        ) / self.session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exporter = MeshExporter(self.output_dir)
        self.captures = []

    def add_capture(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        metadata: dict = None,
    ):
        """Add a capture to the session."""
        capture_id = len(self.captures)
        self.captures.append({
            "points": points,
            "colors": colors,
            "metadata": metadata or {},
            "id": capture_id,
        })
        print(f"Capture {capture_id}: {len(points)} points")

    def merge_captures(self) -> tuple:
        """Merge all captures into single point cloud."""
        if not self.captures:
            return None, None

        all_points = []
        all_colors = []

        for cap in self.captures:
            all_points.append(cap["points"])
            all_colors.append(cap["colors"])

        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        return merged_points, merged_colors

    def export_session(
        self,
        format_type: str = "ply",
        create_mesh: bool = False,
    ) -> Path:
        """Export merged session to file."""
        points, colors = self.merge_captures()
        if points is None:
            print("No captures to export")
            return None

        return self.exporter.export(
            points,
            colors,
            format_type=format_type,
            filename=f"{self.session_name}_merged",
            create_mesh=create_mesh,
        )

    def get_stats(self) -> dict:
        """Get session statistics."""
        total_points = sum(len(c["points"]) for c in self.captures)
        return {
            "session_name": self.session_name,
            "num_captures": len(self.captures),
            "total_points": total_points,
            "output_dir": str(self.output_dir),
        }
