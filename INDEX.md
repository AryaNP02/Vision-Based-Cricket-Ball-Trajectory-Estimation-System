# 📑 Complete Project Index

## 🎯 Quick Links by Purpose

### 📖 I Want to Learn
1. **New to project?** → `README.md`
2. **How to install?** → `docs/SETUP.md`
3. **How to use?** → `docs/USAGE.md`
4. **Need API help?** → `docs/API.md`
5. **Understand structure?** → `PROJECT_STRUCTURE.md`


### 🚀 I Want to Run
1. **Train a model** → `python scripts/train.py`
2. **Run inference** → `python scripts/inference.py`
3. **Preprocess data** → `python scripts/preprocess.py`
4. **See results** → `experiments/results/`
5. **View configs** → `config/*.yaml`

### 💻 I Want to Code
1. **Detect balls** → `from src.detection import BallDetector`
2. **Track balls** → `from src.tracking import BallTracker`
3. **Process videos** → `from src.utils import VideoProcessor`
4. **Export results** → `from src.utils import ResultsExporter`
5. **Full reference** → `docs/API.md`

### 🧪 I Want to Test
1. **Run all tests** → `pytest tests/`
2. **Test detector** → `pytest tests/test_detector.py`
3. **Test tracker** → `pytest tests/test_tracker.py`

### 🛠️ I Want to Configure
1. **Training settings** → `config/training.yaml`
2. **Inference settings** → `config/inference.yaml`
3. **Dataset paths** → `config/dataset.yaml`

### 📊 I Want to Understand
1. **Directory structure** → `PROJECT_STRUCTURE.md`
2. **File organization** → `STRUCTURE.txt`
3. **What's new?** → `MIGRATION.md`
4. **Summary** → `SUMMARY.md`

---

## 📂 Directory Guide

### `/src/` - Source Code
```
src/
├── detection/detector.py     ← Ball detection with YOLO
├── tracking/tracker.py       ← Ball tracking across frames
└── utils/helpers.py          ← Video I/O & CSV export
```
**When to use:** Import these classes in your code

### `/data/` - Data Storage
```
data/
├── raw/                      ← Place videos here
├── processed/                ← Dataset splits (train/val/test)
└── annotations/              ← Additional annotations
```
**When to use:** Store and organize your data

### `/models/` - Model Storage
```
models/
├── pretrained/               ← Download pre-trained weights
└── checkpoints/              ← Save trained models here
```
**When to use:** Store model files

### `/scripts/` - Executable Scripts
```
scripts/
├── train.py                  ← Run: python scripts/train.py
├── inference.py              ← Run: python scripts/inference.py
└── preprocess.py             ← Run: python scripts/preprocess.py
```
**When to use:** Run these to execute tasks

### `/config/` - Configuration
```
config/
├── training.yaml             ← Edit for training settings
├── inference.yaml            ← Edit for inference settings
└── dataset.yaml              ← Edit for dataset paths
```
**When to use:** Modify settings without touching code

### `/experiments/` - Results
```
experiments/
├── logs/                     ← Training logs
├── results/                  ← Inference outputs
└── metrics/                  ← Evaluation metrics
```
**When to use:** Check outputs and results

### `/docs/` - Documentation
```
docs/
├── SETUP.md                  ← Installation help
├── USAGE.md                  ← How-to guide
└── API.md                    ← API reference
```
**When to use:** Learn how to do something

### `/tests/` - Unit Tests
```
tests/
├── test_detector.py          ← Tests for BallDetector
└── test_tracker.py           ← Tests for BallTracker
```
**When to use:** Verify code works correctly

---

## 📄 Documentation Files

### Must Read
| File | Purpose | Read Time |
|------|---------|-----------|
| `README.md` | Overview & quick start | 5 min |
| `docs/SETUP.md` | Installation guide | 5 min |
| `docs/USAGE.md` | How to use | 10 min |

### Should Read
| File | Purpose | Read Time |
|------|---------|-----------|
| `PROJECT_STRUCTURE.md` | Detailed structure | 10 min |
| `docs/API.md` | Code reference | 15 min |

### Nice to Know
| File | Purpose | Read Time |
|------|---------|-----------|
| `MIGRATION.md` | Migrate from old structure | 5 min |
| `SUMMARY.md` | Quick summary | 3 min |
| `STRUCTURE.txt` | Visual tree | 2 min |

---

## 🔑 Key Files by Purpose

### To Run Training
```
1. Edit:  config/training.yaml
2. Run:   python scripts/train.py
3. Check: experiments/logs/
4. Use:   models/checkpoints/best.pt
```

### To Run Inference
```
1. Place: data/raw/*.mp4
2. Edit:  config/inference.yaml (optional)
3. Run:   python scripts/inference.py
4. Check: experiments/results/
```

### To Use as Library
```
1. Install: pip install -r requirements.txt
2. Import:  from src.detection import BallDetector
3. Code:    Create detection/tracking pipeline
4. Test:    python -m pytest tests/
```

### To Understand Code
```
1. Read:    docs/API.md (complete reference)
2. Review:  src/detection/detector.py (class docs)
3. Review:  src/tracking/tracker.py (class docs)
4. Review:  src/utils/helpers.py (utility functions)
```

---

## 🎯 Common Workflows

### Workflow 1: First Time Setup
```
1. Read: README.md
2. Read: docs/SETUP.md
3. Run:  pip install -r requirements.txt
4. Continue: Workflow 2 (Run Inference)
```

### Workflow 2: Run Inference
```
1. Place videos: data/raw/*.mp4
2. Run: python scripts/inference.py
3. Check: experiments/results/videos/
4. View: experiments/results/csv/
```

### Workflow 3: Train Model
```
1. Edit: config/training.yaml
2. Prepare: data/processed/{train,val,test}
3. Run: python scripts/train.py
4. Check: experiments/logs/
5. Use: models/checkpoints/best.pt
```

### Workflow 4: Use as Library
```
1. Install: pip install -r requirements.txt
2. Import: from src.detection import BallDetector
3. Code: Create your pipeline
4. Run: python your_script.py
5. Test: pytest tests/
```

### Workflow 5: Extend Code
```
1. Read: docs/API.md
2. Modify: src/detection/ or src/tracking/
3. Add tests: tests/test_*.py
4. Run: pytest tests/
5. Verify: All tests pass
```

---

## 📝 File Matrix

| Task | File | Type | Read/Write |
|------|------|------|-----------|
| Learn basics | README.md | Doc | Read |
| Setup help | docs/SETUP.md | Doc | Read |
| How to use | docs/USAGE.md | Doc | Read |
| API reference | docs/API.md | Doc | Read |
| Training settings | config/training.yaml | Config | Edit |
| Inference settings | config/inference.yaml | Config | Edit |
| Dataset paths | config/dataset.yaml | Config | Edit |
| Run training | scripts/train.py | Script | Execute |
| Run inference | scripts/inference.py | Script | Execute |
| Run preprocessing | scripts/preprocess.py | Script | Execute |
| Write detection code | src/detection/detector.py | Code | Import |
| Write tracking code | src/tracking/tracker.py | Code | Import |
| Write utility code | src/utils/helpers.py | Code | Import |
| Test detector | tests/test_detector.py | Code | Run |
| Test tracker | tests/test_tracker.py | Code | Run |
| Place videos | data/raw/ | Data | Write |
| Place images | data/processed/ | Data | Write |
| Store models | models/checkpoints/ | Model | Write |
| Check logs | experiments/logs/ | Output | Read |
| Check results | experiments/results/ | Output | Read |

---

## 🔍 Search Guide

### Looking for...
- **How to train?** → `docs/USAGE.md` → "Training" section
- **How to run inference?** → `docs/USAGE.md` → "Inference Options"
- **Code examples?** → `docs/USAGE.md` → "Python API Usage"
- **API documentation?** → `docs/API.md` → Class reference
- **Configuration examples?** → `config/*.yaml` files
- **Test examples?** → `tests/test_*.py` files
- **Video processing?** → `src/utils/helpers.py`
- **Ball detection?** → `src/detection/detector.py`
- **Ball tracking?** → `src/tracking/tracker.py`

---

## ⚡ Quick Commands

```bash
# Setup
pip install -r requirements.txt

# Training
python scripts/train.py
python scripts/train.py --config config/training.yaml

# Inference
python scripts/inference.py
python scripts/inference.py --video input.mp4
python scripts/inference.py --config config/inference.yaml

# Preprocessing
python scripts/preprocess.py --source raw --output data/processed

# Testing
pytest tests/
pytest tests/test_detector.py
pytest tests/ --cov=src

# View files
cat config/training.yaml
cat config/inference.yaml
cat docs/API.md

# Run Python
python -c "from src.detection import BallDetector; print('Import OK')"
```

---

## 📞 FAQ Quick Links

### Q: How do I install?
A: Read `docs/SETUP.md`

### Q: How do I use it?
A: Read `docs/USAGE.md`

### Q: How do I run inference?
A: See `docs/USAGE.md` → "Inference Options"

### Q: How do I train?
A: See `docs/USAGE.md` → "Training"

### Q: What's the API?
A: See `docs/API.md`

### Q: Where do I put videos?
A: `data/raw/`

### Q: Where are results?
A: `experiments/results/`

### Q: Where are logs?
A: `experiments/logs/`

### Q: How do I modify code?
A: Edit `src/` files and follow `docs/API.md`

### Q: How do I test?
A: Run `pytest tests/`

### Q: How do I understand the structure?
A: Read `PROJECT_STRUCTURE.md`

### Q: How do I migrate from old?
A: Read `MIGRATION.md`

---

## 📊 File Statistics

```
Total Files:        25+
Total Directories:  24
Documentation:      6 files
Source Code:        7 files
Scripts:            3 files
Config:             3 files
Tests:              3 files
Setup:              3 files
```

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] `README.md` is readable
- [ ] `docs/SETUP.md` has clear instructions
- [ ] `docs/USAGE.md` has examples
- [ ] `docs/API.md` has all classes
- [ ] `config/*.yaml` files exist
- [ ] `scripts/*.py` files are executable
- [ ] `src/` modules can be imported
- [ ] `tests/` can run
- [ ] `data/` directories exist
- [ ] `models/` directories exist
- [ ] `experiments/` directories exist

---

## 🎓 Learning Path

### Beginner (30 min)
1. Read: `README.md` (5 min)
2. Read: `docs/SETUP.md` (5 min)
3. Run: `pip install -r requirements.txt` (5 min)
4. Run: `python scripts/inference.py` (10 min)

### Intermediate (1 hour)
1. Read: `docs/USAGE.md` (20 min)
2. Edit: `config/inference.yaml` (10 min)
3. Run: Multiple inference tests (20 min)
4. Check: `experiments/results/` (10 min)

### Advanced (2+ hours)
1. Read: `docs/API.md` (30 min)
2. Read: `src/` code files (30 min)
3. Write: Custom code using API (30 min)
4. Test: Run `pytest tests/` (15 min)

---

## 🚀 Ready to Go!

Everything is organized and documented. Pick a file above and get started!

**Recommended first steps:**
1. Read this file (you're doing it!)
2. Read `README.md`
3. Read `docs/SETUP.md`
4. Follow setup instructions
5. Run your first inference!

---

**Project Location:**
```
/home/aryanprajapati/my_folder/sem_3/resume_project/Cricket-Ball-Tracking-Refactored/
```

**Happy coding!** 🎯
