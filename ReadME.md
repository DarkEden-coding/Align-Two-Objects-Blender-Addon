# Align Mesh to Point Cloud — Blender add‑on

Blender add‑on to align a 3D scan mesh to a point‑cloud produced by
Blender's camera tracking. Supports manual vertex correspondences plus
ICP refinement to compute a best‑fit similarity transform. It will rotate, scale, and move the object to best fit the tracking point-cloud.

Quick features:
- Manual vertex‑pair correspondences (source → target).
- Least‑squares similarity fit (SVD) with optional uniform scaling.
- ICP‑style iterative refinement with trimming & sampling for robustness.
- Works in world coordinates and on evaluated meshes (modifiers).

### Quick start (step-by-step)
1. Get the most up-to-date release zip file from the [releases page](https://github.com/DarkEden-coding/Align-3d-Scan-Blender/releases/latest).
2. Install the local addon with the following button, select the zip file after that:
<img width="486" height="123" alt="image" src="https://github.com/user-attachments/assets/13e380ed-29c5-47fa-b9b8-dd4622a2d134" />

3. In the 3D View, open the Sidebar (`N`) and switch to the `Align` tab.
4. Select your **Source** object (the one you want to move) and your **Target** object (the one you want to align to) from the dropdowns.
5. To pick a vertex index from a mesh:
   - Select the mesh and enter Edit Mode.
   - Select a vertex (or make the vertex the active element), then click `Pick Src` (for the source vertex) or `Pick Tgt` (for the target vertex).
   - The picked index will appear in the `Src Index` / `Tgt Index` boxes.
6. After choosing both indices, click `Add Pair`. Repeat to add at least **3** pairs (3 non-collinear points are required for a stable 3D similarity solution).
7. When you have added enough pairs, click **Align Source To Target** to compute and apply the best-fit transform.
8. Optionally, run **Refine (ICP)** to iteratively match the source mesh vertices to the nearest target mesh vertices and further reduce alignment error.

### UI & Settings explained
- `Source`: The object that will be transformed (moved/rotated/scaled).
- `Target`: The object you want the source to align to.
- `Src Index` / `Tgt Index`: The currently-selected vertex indices captured from the active mesh in Edit Mode. Use `Pick Src` / `Pick Tgt` while in Edit Mode to set these values.
- `Pick Src` / `Pick Tgt`: Capture the active or selected vertex index from the active mesh in Edit Mode and assign it to the source or target index respectively.
- `Add Pair`: Store the current `Src Index` and `Tgt Index` as a correspondence pair. Pairs are shown in the list below the controls.
- Pair list: Displays stored correspondences as `SourceName[index] → TargetName[index]`. Use the remove and clear buttons to manage them.
- `Use Uniform Scale`: If enabled, the alignment estimates a single uniform scale factor in addition to rotation and translation. Disable for scale-preserving alignment.
- `Use Modifiers`: When enabled, the add-on will evaluate the object with modifiers applied (via the dependency graph) when collecting points — useful if your meshes rely on modifiers that change the vertex positions.
- `Timeout (s)`: Maximum time in seconds allowed for alignment/refinement operations before they abort with an error.

Refine (ICP) options
- `Max Iterations`: The maximum number of ICP iterations to run before stopping.
- `Tolerance`: Convergence threshold. If improvement in mean matching error between iterations is smaller than this value, ICP stops.
- `Max Distance` (Max Distance to match): Matches with distance greater than this threshold are ignored. Set to `0.0` to disable thresholding.
- `Trim Fraction`: Fraction of worst matches to discard each iteration (robust trimming). Range is 0.0 to 0.49. Use trimming to reduce the influence of outliers.
- `Sample Count`: Limit the number of source vertices considered each iteration. `0` means use all vertices. Increasing this can speed up ICP on large meshes.

### Tips and troubleshooting
- You must provide at least **3** valid pairs to run the one-shot alignment. If you see an error about pairs, check that the indices are correct and belong to the selected objects.
- Use `Use Modifiers` when your target or source relies on modifiers (subdivision, boolean, etc.) — otherwise, the add-on samples raw mesh data.
- The `Refine (ICP)` step is iterative and may take longer on dense meshes; use `Sample Count` and `Max Distance` to speed it up and reduce spurious matches.
- If a **Timeout** error occurs, either raise the timeout threshold or lower the max distance and sample count.
- Each alignment operation is undoable via Blender's normal undo system (`Ctrl+Z`).

### Developer notes
- The add-on computes transforms in world space and applies them to `source.matrix_world`.
- The similarity transform is computed using a least-squares SVD approach and supports optional uniform scaling.

If you have any issues, suggestions, or questions, feel free to open an issue!
