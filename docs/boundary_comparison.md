# FIRESYNC — Bandipur Boundary Comparison

## Comparison status
Completed — authoritative boundary selection unresolved.

## Sources
1. NTCA PA/TR KML
2. Esri India Living Atlas Wildlife Protected Areas 2022

Both returned a Bandipur National Park polygon for Karnataka.

## Geometry measurements

| Metric | NTCA | Esri |
|---|---:|---:|
| Reported area (km2) | 874.2 | 874.2 |
| Calculated geodesic area (km2) | 896.8753 | 933.2510 |
| Polygon components | 1 | 1 |
| Interior holes | 2 | 0 |
| Geometry valid | Yes | Yes |

## Overlay comparison

- Intersection area: 833.6132 km2
- Union area: 996.5130 km2
- Symmetric difference: 162.8998 km2
- Intersection-over-union: 83.6530%
- Boundary Hausdorff distance: 7101.089 metres
- Topologically equal: No
- Coordinate-equal: No

## Satellite observation comparison

Sample: 2025-03-01 to 2025-03-05

- Total observations: 15
- Inside NTCA geometry: 0
- Inside Esri geometry: 0
- Classification changes: 0

This result does not prove either boundary is authoritative.

## Decision

1. Retain the NTCA geometry as provisional.
2. Retain the Esri geometry as an alternative comparison source.
3. Do not merge or average the geometries.
4. Do not modify satellite coordinates to force polygon membership.
5. Seek legally and geographically traceable boundary data.
6. Continue regional analysis using the explicitly defined Bandipur
   landscape rectangle, not an assumed reserve polygon.

## Analytical landscape rectangle

- West: 76.15
- South: 11.55
- East: 76.91
- North: 11.96
- CRS: EPSG:4326

This rectangle is an analytical region of interest, not an
official National Park or Tiger Reserve boundary.

## Provenance

The NTCA DSS page explicitly disclaims guarantees concerning
the accuracy, completeness and currency of its KML.

The Esri public layer reports the feature attributes and polygon
geometry, but its item information does not establish the
underlying legal boundary source or methodology.

## Output evidence

- data/interim/boundary_geometry_comparison.json
- data/interim/boundary_point_comparison.csv
- data/interim/spatial_quality_control.csv
