# milkyEdgeFlowTools

English | [日本語](README_ja.md)

Edge flow tools for Blender 5.0+. Currently ships one operator:

**Relax Crossing Flows** — resamples selected edge loops on a fitted curve so
that the edge loops crossing them relax. The shape of the selected loop is
preserved by the curve; only the distribution of its vertices changes, sliding
each vertex to where the crossing flows naturally want to arrive.

## How it works

- Selected edges are split into chains (open, partial, and closed loops all
  work; branched selections are skipped with a warning).
- Each chain is fitted with a centripetal Catmull-Rom spline that passes
  through every original vertex.
- For each vertex, the incoming direction of the crossing edge loops is
  extrapolated onto the curve to find the relaxed position.
- Vertices whose crossing edges define the object's shape (mesh boundaries,
  sharp creases, edges marked Sharp) are pinned, and their influence falls
  off smoothly with distance.

## Installation

1. Download `milky_edge_flow_tools-x.y.z.zip` from
   [Releases](../../releases).
2. In Blender: `Edit > Preferences > Get Extensions >` (dropdown) `> Install
   from Disk...` and pick the zip.

## Usage

1. In Edit Mode, select one or more edge loops (partial loops are fine).
2. Right-click and choose `milkyEdgeFlowTools > Relax Crossing Flows`.
3. Tune the result in the Redo panel:

| Option | Default | Description |
|---|---|---|
| Factor | 1.0 | Blend between original and relaxed positions |
| Side Blend | 0.0 | Blend between the flow of the side with more rings (0) and fewer rings (1) |
| Face Angle Limit | 90° | Crossing edges whose faces meet at this interior angle or less are treated as shape-defining and pinned |
| Stiffness | 1.0 | Smoothness of the redistribution; higher values spread the influence of pinned vertices further |

## Development

Requires [uv](https://docs.astral.sh/uv/) only — no Blender installation
needed. Integration tests run against the official
[bpy wheel](https://pypi.org/project/bpy/), and builds use a vendored copy
of Blender's official `blender_ext.py` (`tools/`). Tasks are defined in
`pyproject.toml` (poethepoet).

| Command | Description |
|---|---|
| `uv run poe test` | All tests (unit + bpy integration) |
| `uv run poe test-core` | Unit tests only |
| `uv run poe test-blender` | Integration test against the bpy wheel |
| `uv run poe test-blender-app` | Same test inside a real Blender (optional; set `BLENDER` if not at the default path) |
| `uv run poe validate` | Validate the extension manifest |
| `uv run poe build` | Build the distributable zip into `dist/` |

The full specification lives in [requirements.md](requirements.md)
(Japanese). Releases are automated with release-please; commit messages
follow [Conventional Commits](https://www.conventionalcommits.org/).

## License

GPL-3.0-or-later
