# Contributing to PhishGuard

We welcome contributions! This guide helps you get started with contributing to the PhishGuard project.

## 🚀 Quick Start for Contributors

### Development Environment Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/phishguard-scaffold.git
cd phishguard-scaffold

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests to verify setup
python -m pytest tests/ -v  # When tests are available
```

### Running the Framework

```bash
# Generate demo data
python scripts/generate_demo_data.py --tweets 5000 --users 1000

# Run basic training
python -m training.train --config configs/config.yaml

# Run with MLflow tracking
python -m training.train_mlflow --config configs/mlflow_config.yaml
```

## 🛠️ Development Guidelines

### Code Style

We maintain high code quality standards:

- **Consistent formatting**: Follow the existing code style
- **Clear documentation**: Add docstrings to all functions
- **Type hints**: Include type annotations where helpful
- **Error handling**: Graceful error handling and informative messages

```bash
# Format code (if Black is available)
black training/ models/ propagation/ scripts/

# Check for issues
python -m flake8 training/ models/ propagation/
```

### Testing Requirements

- Test your changes with both demo and real data
- Ensure the training pipeline runs without errors
- Verify that new features integrate with existing components
- Include example usage in docstrings

### Documentation

- Update docstrings for any modified functions
- Add usage examples for new features
- Update relevant README sections
- Include configuration examples

## 📋 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:
- Operating system and Python version
- Complete error traceback
- Steps to reproduce the issue
- Configuration file used
- Expected vs actual behavior

### ✨ Feature Requests

For new features:
- Describe the use case and motivation
- Explain how it fits with the existing architecture
- Consider backward compatibility
- Provide implementation ideas if you have them

### 🔧 Code Contributions

#### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding guidelines
3. **Test thoroughly** with both demo and real data
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

#### Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: add new intervention strategy based on influence centrality
fix: resolve memory leak in graph construction
docs: update API documentation for model loading
refactor: simplify adversarial training logic
perf: optimize IC spread simulation for large graphs
```

## 🏗️ Project Architecture

Understanding the codebase structure:

```
phishguard-scaffold/
├── models/              # Model implementations (LLaMA, DistilBERT)
├── training/            # Training loops, adversarial training, optimization
├── propagation/         # Graph algorithms, IC simulation, intervention
├── data/               # Data loading, preprocessing, formatting
├── eval/               # Evaluation metrics and analysis
├── scripts/            # Utility scripts for data collection, generation
├── configs/            # Configuration files for different setups
└── docs/               # Documentation (guides, API reference)
```

## 🎯 Areas Where We Need Help

### High Priority
- [ ] **Performance optimization** for large social networks (100k+ nodes)
- [ ] **Memory efficiency** improvements for limited resource environments
- [ ] **Real-time inference** capabilities for production deployment
- [ ] **Multi-language support** beyond English text

### Research Enhancements
- [ ] **Alternative diffusion models** beyond Independent Cascade
- [ ] **Novel intervention strategies** using graph neural networks
- [ ] **Advanced adversarial training** methods
- [ ] **Federated learning** capabilities for distributed deployment

### Documentation & Usability
- [ ] **Interactive tutorials** with Jupyter notebooks
- [ ] **Video demonstrations** of key features
- [ ] **Performance benchmarks** on various datasets
- [ ] **Integration examples** with popular platforms

### Testing & Quality
- [ ] **Comprehensive test suite** for all components
- [ ] **Continuous integration** setup
- [ ] **Performance regression tests**
- [ ] **Documentation quality checks**

## 🧪 Research Contributions

For academic contributors:

### Standards for Research Features
- Include comprehensive experimental validation
- Compare against relevant baseline methods
- Provide statistical significance analysis
- Document assumptions and limitations
- Include reproducibility information

### Code Quality for Research
- Make research code production-ready when possible
- Include configuration files for experiments
- Provide clear documentation of novel algorithms
- Add extensive comments for complex mathematical operations

## 💡 Getting Help

- **Questions about usage**: Check the documentation in `docs/`
- **Technical discussions**: Open a GitHub Discussion
- **Bug reports**: Create an issue with the bug template
- **Feature requests**: Create an issue with the feature template
- **Urgent issues**: Tag maintainers in your issue

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- The CONTRIBUTORS.md file (when created)
- Release notes for significant contributions  
- Research paper acknowledgments (for research contributions)
- GitHub contributor statistics

---

Thank you for helping improve PhishGuard! Your contributions make this framework better for researchers and practitioners worldwide. 🛡️

For questions or guidance, don't hesitate to reach out through GitHub Issues or Discussions.
