# QuantUI Tutorials

**Welcome to the QuantUI tutorial notebooks!**

These interactive tutorials guide you through quantum chemistry calculations step-by-step. Each tutorial focuses on specific concepts and includes:

- 🎓 **Learning objectives** - What you'll master
- 📝 **Pre-filled code** - Ready to run
- ✅ **Checkpoints** - Verify your understanding
- 💡 **Explanations** - Why things work the way they do
- 🎯 **Expected results** - Know if you got it right

---

## Tutorial Series

### 1. [First Calculation](01_first_calculation.ipynb) - **Start Here!**
**Time:** ~15–20 minutes
**Difficulty:** Beginner

Your first quantum chemistry calculation!
- Run a water molecule calculation
- Interpret basic results
- Understand SCF convergence

**Perfect for:** Complete beginners, first-time users

---

### 2. [Basis Set Study](02_basis_set_study.ipynb)
**Time:** ~30–35 minutes
**Difficulty:** Intermediate

Compare different basis sets on the same molecule.
- Run H₂O with STO-3G, 6-31G, and cc-pVDZ
- Compare energies and computational costs
- Understand accuracy vs. speed trade-offs
- See basis set convergence

**Perfect for:** Understanding what basis sets do

---

### 3. [Multiplicity & Radicals](03_multiplicity_radicals.ipynb)
**Time:** ~25–30 minutes
**Difficulty:** Intermediate

Learn about spin states with oxygen molecule.
- Calculate O₂ as singlet (wrong!)
- Calculate O₂ as triplet (correct!)
- Understand multiplicity = 2S + 1
- Learn when to use UHF vs RHF

**Perfect for:** Understanding open-shell systems

---

### 4. [Charged Species](04_charged_species.ipynb)
**Time:** ~20–25 minutes
**Difficulty:** Intermediate

Work with ions and charged molecules.
- Calculate NH₃ (neutral)
- Calculate NH₄⁺ (cation)
- Calculate NH₂⁻ (anion)
- Understand protonation effects

**Perfect for:** Understanding charge effects

---

### 5. [Comparing Results](05_comparing_results.ipynb)
**Time:** ~25–30 minutes
**Requires:** `pandas` (`pip install pandas`; not a QuantUI dependency)
**Difficulty:** Intermediate

Analyze and compare multiple calculations.
- Load multiple job results
- Create comparison tables
- Visualize energy trends
- Export data for reports

**Perfect for:** Data analysis and reporting

---

## How to Use

### 📖 For Self-Study

1. **Start with Tutorial 1** - Even if you're experienced
2. **Work sequentially** - Each builds on previous concepts
3. **Read explanations** - Don't just run cells
4. **Try variations** - Experiment with different molecules
5. **Check solutions** - Compare your results

### 🎓 For Classroom Use

Instructors can:
- Assign specific tutorials as homework
- Use in lab sessions with guidance
- Modify molecules for assignments
- Add questions/discussion points

### ⏱️ Suggested Schedule

- **Week 1:** Tutorial 1 (First Calculation)
- **Week 2:** Tutorial 2 (Basis Sets)
- **Week 3:** Tutorial 3 (Multiplicity) + Tutorial 4 (Charged Species)
- **Week 4:** Tutorial 5 (Comparing Results) + Final Project

---

## Prerequisites

- Access to QuantUI main notebook
- Basic understanding of:
  - Chemical formulas
  - Atomic structure
  - Basic quantum concepts (orbitals, electrons)

---

## Tips for Success

✅ **Run cells in order** - Don't skip ahead
✅ **Read all text** - Explanations are important
✅ **Experiment** - Try changing molecules
✅ **Use checkpoints** - Verify before proceeding
✅ **Ask questions** - Contact your instructor

---

## Troubleshooting

**Problem:** Jobs fail immediately
**Solution:** Check your molecule setup in checkpoint cells

**Problem:** Can't see visualizations
**Solution:** Ensure py3Dmol is installed (see main notebook setup)

**Problem:** Different energy than expected
**Solution:** Small differences are okay - check units and significant figures

---

## Next Steps

After completing these tutorials:
1. Use the main app — [molecule_computations.ipynb](../molecule_computations.ipynb), or `quantui run app` — for your own calculations
2. Use the app's **History**, **Analysis** and **Compare** tabs to revisit and compare results
3. Experiment with molecules from your coursework

---

## Contributing

Found an error? Have suggestions?
- Create an issue on GitHub
- Contact your instructor
- Propose improvements

**Happy calculating!** 🧪⚛️
