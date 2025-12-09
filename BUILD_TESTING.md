# Build Testing and CI/CD

## Problem Summary

ARM64 cross-compilation was failing with the error:
```
/usr/aarch64-unknown-linux-gnu/bin/ld.bfd: cannot find -lopenblas
```

### Root Cause

The workflow was using `blas-system` feature for Linux builds, which attempts to link against system OpenBLAS libraries. During ARM64 cross-compilation:

1. The manylinux container runs on x86_64 architecture
2. `before-script-linux` only installed x86_64 OpenBLAS packages
3. When linking ARM64 binaries, the linker couldn't find ARM64 OpenBLAS libraries

## Solution

Reverted to using `blas-static` feature for Linux (matching the previous working configuration):

### Changes Made

1. **release.yml** - Linux builds:
   - Changed from `blas-system` → `blas-static`
   - Added environment variables:
     - `OPENBLAS_STATIC=1`
     - `OPENBLAS_NO_DYNAMIC_ARCH=1`
     - `OPENBLAS_TARGET=ARMV8` (for aarch64) or `GENERIC` (for x86_64)
   - Updated dependencies: gfortran, make, cmake, perl (needed to compile OpenBLAS)

2. **release.yml** - macOS builds:
   - Added OpenBLAS environment variables for consistency
   - Already using `blas-static` (no change needed)

3. **pyproject.toml**:
   - Updated default to `blas-static` for consistency

## Why This Works

### Static Linking Approach

- **Compiles OpenBLAS from source** for the target architecture
- **Cross-compilation friendly**: Build system downloads source and compiles for ARM64
- **Self-contained wheels**: Bundles LAPACK support without external dependencies
- **Proven approach**: Matches commit 7bbe34c which worked successfully

### GitHub Actions Environment

The manylinux containers in GitHub Actions:
- Have working SSL certificates (unlike local sandboxed test environments)
- Can download OpenBLAS source via HTTPS
- Have all necessary build tools (make, cmake, perl, gfortran)
- Support cross-compilation with appropriate toolchains

## Testing

### Automated Testing

Created `.github/workflows/test-build.yml` that:
- Tests both x86_64 and aarch64 on Linux
- Tests both targets on macOS
- Includes detailed logging of:
  - Installed dependencies
  - Environment variables
  - Build process
  - Wheel verification
  - Import tests
  - Shared library dependencies

### Triggering Tests

```bash
# Test builds run automatically on push to claude/* branches
git push origin claude/your-branch-name

# Or trigger manually via GitHub Actions UI:
# Actions → Test Build → Run workflow
```

### Local Testing Limitations

Local ARM64 cross-compilation testing is blocked by:
- SSL certificate verification issues in sandboxed environments
- These issues don't occur in GitHub Actions with proper certificates

## Verification Steps

After pushing changes:

1. **Check GitHub Actions**:
   - Go to: https://github.com/AlekJaworski/nn_exploring/actions
   - Look for "Test Build" workflow run
   - Verify both x86_64 and aarch64 Linux builds succeed

2. **Review Build Logs**:
   - Check "Build wheels" step shows OpenBLAS compilation
   - Verify no "cannot find -lopenblas" errors
   - Confirm wheels are created in `dist/` directory

3. **Validate Wheels**:
   - Check wheel tags match target architectures
   - Verify `.so` files are included
   - Test import succeeds

## Comparison with Previous Versions

### Commit 7bbe34c (Working)
- Linux: `blas-static` with gfortran ✅
- macOS: `blas-static` ✅
- Result: All builds successful

### Commit e2587d0 (Broken)
- Linux: `blas-system` with system OpenBLAS ❌
- macOS: `blas-static` ✅
- Result: ARM64 Linux failed with "cannot find -lopenblas"

### Current Fix (This Branch)
- Linux: `blas-static` with gfortran ✅
- macOS: `blas-static` with env vars ✅
- Expected: All builds successful (matches 7bbe34c approach)

## Future Improvements

If `blas-static` compilation becomes too slow, consider:

1. **Caching**: Use GitHub Actions cache for compiled OpenBLAS
2. **Pre-built binaries**: Vendor pre-compiled ARM64 OpenBLAS libraries
3. **Matrix optimization**: Build sequentially instead of parallel to manage resources
