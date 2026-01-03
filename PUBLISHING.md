# Publishing py-scaffold to PyPI

This guide explains how to publish the py-scaffold package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/
2. **Test PyPI Account** (recommended for testing): Create an account at https://test.pypi.org/
3. **API Token**: Generate an API token from your PyPI account settings

## Step 1: Create API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Give it a descriptive name (e.g., "py-scaffold-upload")
4. Set the scope to "Entire account" or specific to "py-scaffold" after first upload
5. Copy the token (starts with `pypi-`)
6. **Save it securely** - you won't be able to see it again

## Step 2: Configure PyPI Credentials

### Option A: Using .pypirc file (Recommended)

Create/edit `~/.pypirc` with your credentials:

```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Make sure to secure the file:
```bash
chmod 600 ~/.pypirc
```

### Option B: Set Environment Variable

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE
```

## Step 3: Test on Test PyPI (Recommended)

Before publishing to the real PyPI, test on Test PyPI:

```bash
# Activate virtual environment
source venv/bin/activate

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --no-deps py-scaffold
```

## Step 4: Publish to PyPI

Once you've verified everything works on Test PyPI:

```bash
# Activate virtual environment
source venv/bin/activate

# Upload to PyPI
twine upload dist/*

# Or if prompted for credentials, use:
# Username: __token__
# Password: pypi-YOUR_TOKEN_HERE
```

## Step 5: Verify Publication

1. Check your package page: https://pypi.org/project/py-scaffold/
2. Test installation:
```bash
# In a new environment
pip install py-scaffold

# Test the CLI
py-scaffold --help
```

## Publishing New Versions

When you want to publish a new version:

1. **Update version number** in:
   - `pyproject.toml`
   - `src/py_scaffold/__init__.py`
   - `CHANGELOG.md` (add new version section)

2. **Clean old builds**:
```bash
rm -rf dist/ build/ src/*.egg-info
```

3. **Build new distribution**:
```bash
source venv/bin/activate
python -m build
```

4. **Verify the build**:
```bash
twine check dist/*
```

5. **Upload to PyPI**:
```bash
twine upload dist/*
```

## Troubleshooting

### Error: File already exists

PyPI doesn't allow re-uploading the same version. You must:
- Increment the version number
- Rebuild the package
- Upload the new version

### Error: Invalid or non-existent authentication

- Double-check your API token
- Ensure username is `__token__` (with double underscores)
- Make sure the token starts with `pypi-`

### Error: Package name already taken

If "py-scaffold" is already taken, you'll need to:
- Choose a different name in `pyproject.toml`
- Update all references to the package name
- Rebuild and upload

## Security Best Practices

1. **Never commit API tokens** to version control
2. **Use scoped tokens** when possible (project-specific)
3. **Rotate tokens** periodically
4. **Use .pypirc with restricted permissions** (chmod 600)
5. **Consider using keyring** for credential management

## Automated Publishing with GitHub Actions

For automated publishing on releases, add this workflow to `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Then add your PyPI API token as a GitHub secret named `PYPI_API_TOKEN`.

## Current Build Status

The package has been built and is ready for publishing:
- Location: `dist/`
- Files:
  - `py_scaffold-0.1.0-py3-none-any.whl` (wheel)
  - `py_scaffold-0.1.0.tar.gz` (source distribution)
- Status: ✓ Passed twine check

You can now upload these files to PyPI using the instructions above!
