"""Test QuantUI PlotlyMol Integration"""

from quantui import Molecule, is_visualization_available, visualize_molecule

print("Testing QuantUI PlotlyMol Integration")
print("=" * 50)

# Check if visualization is available
print(f"\n1. Visualization available: {is_visualization_available()}")

# Create a test molecule (water)
mol = Molecule(
    atoms=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    charge=0,
    multiplicity=1,
)

print(f"2. Created molecule: {mol.get_formula()}")
print(f"3. Electron count: {mol.get_electron_count()}")
print("4. XYZ string output:")
print(mol.to_xyz_string())

# Test visualization
print("\n5. Creating 3D visualization...")
try:
    fig = visualize_molecule(mol, backend="plotlymol", mode="ball+stick")
    print(f"   - Figure created: {fig is not None}")
    print(f"   - Number of traces: {len(fig.data)}")
    print(f"   - Figure title: {fig.layout.title.text}")
    print("\n" + "=" * 50)
    print("SUCCESS: All tests passed!")
    print("=" * 50)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
