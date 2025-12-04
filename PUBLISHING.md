# Publishing to PyPI

This guide explains how to publish `mgcv-rust` to PyPI for cross-platform distribution.

## Architecture Strategy

We use system OpenBLAS for all builds:

- Uses `openblas-system` (requires system OpenBLAS to be installed)

## Supported Platforms

The GitHub Actions workflow builds wheels for:

| Platform | Architectures |
|----------|---------------|
| Linux    | x86_64, aarch64 (ARM64) |
| macOS    | x86_64 (Intel), aarch64 (Apple Silicon) |

## Manual Publishing

### 1. Prerequisites

```bash
pip install maturin
```

### 2. Update Version

Edit `pyproject.toml` and `Cargo.toml` to bump the version number.

### 3. Build Wheels

For your current platform:
```bash
maturin build --release --features python,blas
```

For multiple Python versions:
```bash
maturin build --release --features python,blas --interpreter python3.8 python3.9 python3.10 python3.11 python3.12
```

Wheels will be in `target/wheels/`.

### 4. Test on TestPyPI

```bash
# Upload to test repository
maturin publish --repository testpypi

# Test installation
pip install --index-url https://test.pypi.org/simple/ mgcv-rust
python -c "import mgcv_rust; print('Success!')"
```

### 5. Publish to PyPI

```bash
maturin publish
```

You'll be prompted for your PyPI credentials or API token.

## Automated Publishing (Recommended)

### Setup

1. **Create PyPI API Token**:
   - Go to https://pypi.org/manage/account/
   - Create a new API token (scope: entire account or just this project)
   - Copy the token (starts with `pypi-`)

2. **Add Token to GitHub Secrets**:
   - Go to your repo → Settings → Secrets and variables → Actions
   - Create new secret: `PYPI_API_TOKEN` = your token

3. **Update Metadata**:
   - Edit `pyproject.toml`: update author name, email, and repository URLs

### Trigger a Release

1. **Create a Git Tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Create GitHub Release**:
   - Go to your repo → Releases → Create a new release
   - Choose the tag you just created
   - Fill in release notes
   - Publish release

3. **Wait for Build**:
   - GitHub Actions will automatically build wheels for all platforms
   - Check the Actions tab for progress
   - Takes ~20-30 minutes

4. **Verify**:
   ```bash
   pip install mgcv-rust
   python -c "import mgcv_rust; print('Success!')"
   ```

## Development Builds

For fast iteration during development:

```bash
# Install system OpenBLAS first
# Ubuntu/Debian: sudo apt-get install libopenblas-dev
# macOS: brew install openblas

# Build with maturin
maturin develop --features python,blas

# Or with cargo
cargo build --release --features python,blas
```

## Troubleshooting

### Build Fails with OpenBLAS Errors

If build fails due to OpenBLAS:
1. Ensure system OpenBLAS is installed (`libopenblas-dev` on Ubuntu, `brew install openblas` on macOS)
2. Check you have build tools: `gcc`, `gfortran`, `make`
3. Check the OpenBLAS linking logs

### Missing Wheels for a Platform

The GitHub Actions workflow might fail for specific platforms. Check:
1. The Actions logs for that platform
2. Whether that platform has the required build tools
3. Update the workflow if platform-specific config is needed

## Alternative: Manual Cross-Platform Builds

If you can't use GitHub Actions:

### Linux (Docker)

```bash
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release --features python,blas
```

### macOS Cross-Compile

```bash
# For Apple Silicon from Intel Mac
rustup target add aarch64-apple-darwin
maturin build --release --features python,blas --target aarch64-apple-darwin
```

## Wheel Naming Convention

Maturin automatically names wheels following PEP standards:

```
mgcv_rust-0.1.0-cp310-cp310-manylinux_2_17_x86_64.whl
  │         │     │      │      │
  │         │     │      │      └─ Platform
  │         │     │      └──────── ABI tag
  │         │     └─────────────── Python version
  │         └───────────────────── Package version
  └─────────────────────────────── Package name
```

Users with matching platform/Python will automatically get the right wheel when running `pip install mgcv-rust`.
