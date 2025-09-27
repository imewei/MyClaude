---
description: Intelligent project cleanup engine with AI-powered file analysis, duplicate detection, and project structure optimization for Python and Julia scientific computing projects
category: project-management-cleanup
argument-hint: [target-path] [--dry-run] [--aggressive] [--type=python|julia|mixed|auto] [--interactive] [--backup] [--git-integration] [--report]
allowed-tools: Read, Write, Edit, MultiEdit, Grep, Glob, TodoWrite, Bash
---

# 🗂️ Revolutionary Project Cleanup Engine (2025 Edition)

Advanced AI-powered project structure analysis and cleanup system with intelligent file duplication detection, obsolete artifact removal, and scientific computing project optimization specialized for Python and Julia ecosystems with comprehensive git integration.

## Quick Start

```bash
# Comprehensive project analysis with interactive cleanup
/clean-project

# Analyze Python scientific computing project
/clean-project --type=python --aggressive --backup

# Julia project optimization with git integration
/clean-project --type=julia --git-integration --interactive

# Mixed Python/Julia project cleanup
/clean-project --type=mixed --comprehensive --report

# Dry-run analysis for large projects
/clean-project . --dry-run --report --comprehensive

# Enterprise-grade cleanup with full safety
/clean-project --aggressive --backup --git-integration --interactive
```

## 🔬 Core Intelligent Project Analysis Engine

### 1. Advanced Multi-Language Project Detection System

```bash
# Comprehensive project ecosystem analysis with scientific computing focus
analyze_project_ecosystem() {
    local target_path="${1:-.}"
    echo "🔍 Initializing Revolutionary Project Analysis Engine..."

    # Initialize analysis environment
    mkdir -p .project_cleanup/{
        analysis_cache,
        file_inventory,
        duplicate_files,
        unused_files,
        obsolete_artifacts,
        empty_directories,
        git_analysis,
        python_specific,
        julia_specific,
        cleanup_plans,
        backups,
        reports,
        metrics
    }

    # Advanced project type detection with confidence scoring
    detect_project_type() {
        echo "  🌐 Detecting project type and scientific computing stack..."

        local project_types=()
        local confidence_scores=()
        local scientific_indicators=()

        # Python ecosystem detection (Enhanced for Scientific Computing)
        detect_python_ecosystem() {
            echo "    🐍 Analyzing Python scientific computing ecosystem..."

            local python_confidence=0
            local python_features=()

            # Core Python indicators
            if [[ -f "pyproject.toml" ]]; then
                python_confidence=$((python_confidence + 30))
                python_features+=("pyproject.toml")
                echo "      📦 Modern Python project structure detected"
            fi

            if [[ -f "setup.py" ]]; then
                python_confidence=$((python_confidence + 20))
                python_features+=("setup.py")
                echo "      📦 Traditional Python package detected"
            fi

            if [[ -f "requirements.txt" ]]; then
                python_confidence=$((python_confidence + 15))
                python_features+=("requirements.txt")
            fi

            if [[ -f "environment.yml" ]] || [[ -f "conda-environment.yml" ]]; then
                python_confidence=$((python_confidence + 20))
                python_features+=("conda_environment")
                echo "      🔬 Conda scientific environment detected"
            fi

            if [[ -f "Pipfile" ]]; then
                python_confidence=$((python_confidence + 15))
                python_features+=("pipenv")
            fi

            # Scientific computing stack detection
            local scientific_packages=(
                "numpy" "scipy" "pandas" "matplotlib" "seaborn" "plotly"
                "scikit-learn" "tensorflow" "pytorch" "jax" "flax" "optax"
                "jupyter" "ipython" "notebook" "lab"
                "xarray" "polars" "dask" "ray"
                "sympy" "networkx" "igraph"
                "opencv" "pillow" "imageio" "scikit-image"
                "h5py" "zarr" "netcdf4" "pyarrow"
                "numba" "cupy" "cython"
                "astropy" "biopython" "nilearn" "dipy"
                "qiskit" "cirq" "pennylane"
            )

            local scientific_count=0
            for package in "${scientific_packages[@]}"; do
                if grep -r -q "$package" . --include="*.txt" --include="*.toml" --include="*.yml" --include="*.yaml" --include="*.py" 2>/dev/null; then
                    scientific_count=$((scientific_count + 1))
                    scientific_indicators+=("python:$package")
                fi
            done

            if [[ $scientific_count -gt 0 ]]; then
                python_confidence=$((python_confidence + scientific_count * 5))
                echo "      🔬 Scientific computing packages detected: $scientific_count"
            fi

            # Deep learning frameworks
            if grep -r -q -E "(torch|tensorflow|jax|flax)" . --include="*.py" --include="*.txt" --include="*.toml" 2>/dev/null; then
                python_confidence=$((python_confidence + 10))
                python_features+=("deep_learning")
                echo "      🧠 Deep learning frameworks detected"
            fi

            # Jupyter environment detection
            if find . -name "*.ipynb" | head -1 | grep -q .; then
                python_confidence=$((python_confidence + 10))
                python_features+=("jupyter_notebooks")
                echo "      📓 Jupyter notebooks detected"
            fi

            # Python file count
            local python_file_count=$(find . -name "*.py" -type f | wc -l)
            if [[ $python_file_count -gt 0 ]]; then
                python_confidence=$((python_confidence + python_file_count / 2))
                echo "      📄 Python files found: $python_file_count"
            fi

            if [[ $python_confidence -gt 20 ]]; then
                project_types+=("python")
                confidence_scores+=($python_confidence)
                printf "%s\n" "${python_features[@]}" > .project_cleanup/python_specific/detected_features.txt
                echo "    ✅ Python project detected (confidence: $python_confidence%)"
            fi
        }

        # Julia ecosystem detection (Enhanced for Scientific Computing)
        detect_julia_ecosystem() {
            echo "    🟣 Analyzing Julia scientific computing ecosystem..."

            local julia_confidence=0
            local julia_features=()

            # Core Julia indicators
            if [[ -f "Project.toml" ]]; then
                julia_confidence=$((julia_confidence + 40))
                julia_features+=("Project.toml")
                echo "      📦 Julia package project detected"
            fi

            if [[ -f "Manifest.toml" ]]; then
                julia_confidence=$((julia_confidence + 20))
                julia_features+=("Manifest.toml")
                echo "      🔒 Julia environment manifest detected"
            fi

            # Julia scientific computing packages
            local julia_scientific_packages=(
                "DataFrames" "CSV" "Plots" "PlotlyJS" "StatsPlots"
                "Flux" "MLJ" "MLBase" "ScikitLearn"
                "DifferentialEquations" "ModelingToolkit" "Symbolics"
                "LinearAlgebra" "Statistics" "StatsBase" "Distributions"
                "Optim" "JuMP" "Convex"
                "CUDA" "CuArrays" "KernelAbstractions"
                "Images" "ImageView" "FileIO"
                "DSP" "FFTW" "SignalAnalysis"
                "Unitful" "Measurements" "PhysicalConstants"
                "BenchmarkTools" "ProfileView" "StatProfilerHTML"
                "Documenter" "DocumenterTools"
                "IJulia" "Pluto" "PlutoUI"
                "PackageCompiler" "BinaryBuilder"
                "HDF5" "JLD2" "Arrow" "Parquet"
            )

            local julia_scientific_count=0
            if [[ -f "Project.toml" ]]; then
                for package in "${julia_scientific_packages[@]}"; do
                    if grep -q "\"$package\"" Project.toml 2>/dev/null; then
                        julia_scientific_count=$((julia_scientific_count + 1))
                        scientific_indicators+=("julia:$package")
                    fi
                done
            fi

            # Check for packages in Julia files
            for package in "${julia_scientific_packages[@]}"; do
                if grep -r -q "using $package" . --include="*.jl" 2>/dev/null; then
                    julia_scientific_count=$((julia_scientific_count + 1))
                    scientific_indicators+=("julia:$package")
                fi
            done

            if [[ $julia_scientific_count -gt 0 ]]; then
                julia_confidence=$((julia_confidence + julia_scientific_count * 8))
                echo "      🔬 Scientific computing packages detected: $julia_scientific_count"
            fi

            # Julia file structure analysis
            if [[ -d "src" ]] && find src -name "*.jl" | head -1 | grep -q .; then
                julia_confidence=$((julia_confidence + 15))
                julia_features+=("src_structure")
                echo "      📁 Standard Julia src/ structure detected"
            fi

            if [[ -d "test" ]] && [[ -f "test/runtests.jl" ]]; then
                julia_confidence=$((julia_confidence + 10))
                julia_features+=("test_structure")
                echo "      🧪 Julia test structure detected"
            fi

            if [[ -d "docs" ]] && [[ -f "docs/make.jl" ]]; then
                julia_confidence=$((julia_confidence + 8))
                julia_features+=("docs_structure")
                echo "      📚 Julia documentation structure detected"
            fi

            # Julia file count
            local julia_file_count=$(find . -name "*.jl" -type f | wc -l)
            if [[ $julia_file_count -gt 0 ]]; then
                julia_confidence=$((julia_confidence + julia_file_count / 2))
                echo "      📄 Julia files found: $julia_file_count"
            fi

            # Performance analysis files
            if find . -name "benchmark" -type d | head -1 | grep -q .; then
                julia_confidence=$((julia_confidence + 5))
                julia_features+=("benchmarks")
                echo "      ⚡ Julia benchmarks detected"
            fi

            if [[ $julia_confidence -gt 15 ]]; then
                project_types+=("julia")
                confidence_scores+=($julia_confidence)
                printf "%s\n" "${julia_features[@]}" > .project_cleanup/julia_specific/detected_features.txt
                echo "    ✅ Julia project detected (confidence: $julia_confidence%)"
            fi
        }

        # Mixed project detection
        detect_mixed_project() {
            if [[ "${#project_types[@]}" -gt 1 ]]; then
                echo "    🔄 Mixed-language scientific computing project detected"
                project_types+=("mixed")

                # Calculate mixed project confidence
                local total_confidence=0
                for score in "${confidence_scores[@]}"; do
                    total_confidence=$((total_confidence + score))
                done
                confidence_scores+=($total_confidence)

                echo "      📊 Language distribution:"
                for i in "${!project_types[@]}"; do
                    if [[ "${project_types[$i]}" != "mixed" ]]; then
                        echo "        ${project_types[$i]}: ${confidence_scores[$i]}%"
                    fi
                done
            fi
        }

        # Execute detection
        detect_python_ecosystem
        detect_julia_ecosystem
        detect_mixed_project

        # Store results
        printf "%s\n" "${project_types[@]}" > .project_cleanup/detected_project_types.txt
        printf "%s\n" "${confidence_scores[@]}" > .project_cleanup/confidence_scores.txt
        printf "%s\n" "${scientific_indicators[@]}" > .project_cleanup/scientific_indicators.txt

        echo "  ✅ Project ecosystem analysis complete"
        return 0
    }

    # Comprehensive file inventory with intelligent categorization
    create_file_inventory() {
        echo "  📋 Creating comprehensive file inventory..."

        # Initialize file categories
        mkdir -p .project_cleanup/file_inventory/{
            source_code,
            data_files,
            config_files,
            documentation,
            build_artifacts,
            cache_files,
            backup_files,
            temporary_files,
            hidden_files,
            large_files,
            binary_files,
            unknown_files
        }

        # Categorize all files
        find "$target_path" -type f -not -path "./.project_cleanup/*" -not -path "./.git/*" | while read -r file; do
            local basename_file=$(basename "$file")
            local extension="${basename_file##*.}"
            local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")

            # Source code files
            case "$extension" in
                py|jl|ipynb|pyx|pxd|pxi)
                    echo "$file" >> .project_cleanup/file_inventory/source_code/files.txt
                    echo "$file:$file_size" >> .project_cleanup/file_inventory/source_code/files_with_size.txt
                    ;;

                # Data files
                csv|json|jsonl|xml|yaml|yml|toml|h5|hdf5|nc|zarr|parquet|arrow|feather|pkl|pickle|npy|npz|mat|jld2)
                    echo "$file" >> .project_cleanup/file_inventory/data_files/files.txt
                    echo "$file:$file_size" >> .project_cleanup/file_inventory/data_files/files_with_size.txt
                    ;;

                # Configuration files
                cfg|conf|config|ini|env|lock)
                    echo "$file" >> .project_cleanup/file_inventory/config_files/files.txt
                    echo "$file:$file_size" >> .project_cleanup/file_inventory/config_files/files_with_size.txt
                    ;;

                # Documentation
                md|rst|txt|pdf|html|tex|org)
                    echo "$file" >> .project_cleanup/file_inventory/documentation/files.txt
                    echo "$file:$file_size" >> .project_cleanup/file_inventory/documentation/files_with_size.txt
                    ;;

                # Build artifacts
                o|a|so|dylib|dll|exe|whl|tar|gz|zip|egg)
                    echo "$file" >> .project_cleanup/file_inventory/build_artifacts/files.txt
                    echo "$file:$file_size" >> .project_cleanup/file_inventory/build_artifacts/files_with_size.txt
                    ;;

                # Image/media files
                png|jpg|jpeg|gif|svg|pdf|eps|ps|tiff|bmp)
                    echo "$file" >> .project_cleanup/file_inventory/binary_files/images.txt
                    echo "$file:$file_size" >> .project_cleanup/file_inventory/binary_files/images_with_size.txt
                    ;;

                *)
                    # Special file patterns
                    case "$basename_file" in
                        .*DS_Store|Thumbs.db|desktop.ini)
                            echo "$file" >> .project_cleanup/file_inventory/temporary_files/system_junk.txt
                            ;;
                        *.tmp|*.temp|*.bak|*.backup|*.old|*~)
                            echo "$file" >> .project_cleanup/file_inventory/backup_files/files.txt
                            ;;
                        .*)
                            echo "$file" >> .project_cleanup/file_inventory/hidden_files/files.txt
                            ;;
                        *)
                            echo "$file" >> .project_cleanup/file_inventory/unknown_files/files.txt
                            echo "$file:$file_size" >> .project_cleanup/file_inventory/unknown_files/files_with_size.txt
                            ;;
                    esac
                    ;;
            esac

            # Large file detection (>50MB)
            if [[ $file_size -gt 52428800 ]]; then
                echo "$file:$file_size" >> .project_cleanup/file_inventory/large_files/files.txt
            fi
        done

        # Cache file detection (Python/Julia specific)
        find "$target_path" -type f \( -path "*/__pycache__/*" -o -path "*/.pytest_cache/*" -o -path "*/.julia/compiled/*" \) | while read -r cache_file; do
            echo "$cache_file" >> .project_cleanup/file_inventory/cache_files/files.txt
        done

        echo "    ✅ File inventory complete"
    }

    # Execute analysis phases
    detect_project_type
    create_file_inventory

    echo "🎯 Project ecosystem analysis complete!"
    return 0
}
```

### 2. Advanced File Duplication Detection Engine

```bash
# Comprehensive file duplication detection with content analysis
detect_file_duplicates() {
    local target_path="${1:-.}"
    echo "🔍 Executing Advanced File Duplication Detection..."

    # Initialize duplication analysis
    mkdir -p .project_cleanup/duplicate_files/{exact,similar,content_analysis}

    # Exact duplicate detection using content hashing
    detect_exact_file_duplicates() {
        echo "  🎯 Detecting exact file duplicates..."

        # Create content hashes for all files
        find "$target_path" -type f -not -path "./.project_cleanup/*" -not -path "./.git/*" | while read -r file; do
            # Calculate content hash
            content_hash=$(shasum -a 256 "$file" 2>/dev/null | cut -d' ' -f1)
            file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")

            echo "$content_hash:$file_size:$file" >> .project_cleanup/duplicate_files/file_hashes.txt
        done

        # Group files by identical content hash
        sort .project_cleanup/duplicate_files/file_hashes.txt | cut -d: -f1 | uniq -c | while read -r count hash; do
            count=$(echo "$count" | xargs)  # trim whitespace

            if [[ $count -gt 1 ]]; then
                echo "    📊 Found $count identical files with hash $hash"

                # Extract all files with this hash
                grep "^$hash:" .project_cleanup/duplicate_files/file_hashes.txt > ".project_cleanup/duplicate_files/exact/group_${hash}.txt"

                # Show duplicate group details
                echo "      🔄 Duplicate group $hash:" >> .project_cleanup/duplicate_files/exact_duplicates_summary.txt
                while read -r duplicate_entry; do
                    hash_part=$(echo "$duplicate_entry" | cut -d: -f1)
                    size_part=$(echo "$duplicate_entry" | cut -d: -f2)
                    file_part=$(echo "$duplicate_entry" | cut -d: -f3-)
                    echo "        📍 $file_part (${size_part} bytes)" >> .project_cleanup/duplicate_files/exact_duplicates_summary.txt
                done < ".project_cleanup/duplicate_files/exact/group_${hash}.txt"
                echo "" >> .project_cleanup/duplicate_files/exact_duplicates_summary.txt
            fi
        done

        # Count total duplicates
        local total_duplicate_groups=$(ls .project_cleanup/duplicate_files/exact/group_*.txt 2>/dev/null | wc -l)
        if [[ $total_duplicate_groups -gt 0 ]]; then
            echo "    📈 Total duplicate groups found: $total_duplicate_groups"
        else
            echo "    ✅ No exact duplicate files found"
        fi
    }

    # Similar file detection (same name, different locations)
    detect_similar_files() {
        echo "  🎯 Detecting similar files (same name, different paths)..."

        # Group files by basename
        find "$target_path" -type f -not -path "./.project_cleanup/*" -not -path "./.git/*" | while read -r file; do
            basename_file=$(basename "$file")
            echo "$basename_file:$file" >> .project_cleanup/duplicate_files/files_by_basename.txt
        done

        # Find files with same basename in different directories
        sort .project_cleanup/duplicate_files/files_by_basename.txt | cut -d: -f1 | uniq -c | while read -r count basename_name; do
            count=$(echo "$count" | xargs)  # trim whitespace

            if [[ $count -gt 1 ]]; then
                echo "    📊 Found $count files named '$basename_name'"

                # Extract all files with this basename
                grep "^$basename_name:" .project_cleanup/duplicate_files/files_by_basename.txt > ".project_cleanup/duplicate_files/similar/basename_${basename_name//\//_}.txt"

                # Check if they have different content
                local same_content=true
                local first_hash=""

                while read -r similar_entry; do
                    file_path=$(echo "$similar_entry" | cut -d: -f2-)
                    file_hash=$(shasum -a 256 "$file_path" 2>/dev/null | cut -d' ' -f1)

                    if [[ -z "$first_hash" ]]; then
                        first_hash="$file_hash"
                    elif [[ "$file_hash" != "$first_hash" ]]; then
                        same_content=false
                        break
                    fi
                done < ".project_cleanup/duplicate_files/similar/basename_${basename_name//\//_}.txt"

                if [[ "$same_content" == "false" ]]; then
                    echo "      ⚠️  Files with name '$basename_name' have different content:" >> .project_cleanup/duplicate_files/similar_files_summary.txt
                    while read -r similar_entry; do
                        file_path=$(echo "$similar_entry" | cut -d: -f2-)
                        echo "        📍 $file_path" >> .project_cleanup/duplicate_files/similar_files_summary.txt
                    done < ".project_cleanup/duplicate_files/similar/basename_${basename_name//\//_}.txt"
                    echo "" >> .project_cleanup/duplicate_files/similar_files_summary.txt
                fi
            fi
        done
    }

    # Jupyter notebook duplicate detection (special handling)
    detect_notebook_duplicates() {
        echo "  📓 Analyzing Jupyter notebook duplicates..."

        # Find all notebooks
        find "$target_path" -name "*.ipynb" -not -path "./.project_cleanup/*" | while read -r notebook; do
            # Extract notebook without outputs for comparison
            if command -v jq >/dev/null 2>&1; then
                # Remove outputs and execution counts for comparison
                jq 'del(.cells[].outputs) | del(.cells[].execution_count)' "$notebook" 2>/dev/null > ".project_cleanup/duplicate_files/content_analysis/$(basename "$notebook" .ipynb)_normalized.json"

                # Calculate hash of normalized content
                normalized_hash=$(shasum -a 256 ".project_cleanup/duplicate_files/content_analysis/$(basename "$notebook" .ipynb)_normalized.json" 2>/dev/null | cut -d' ' -f1)
                echo "$normalized_hash:$notebook" >> .project_cleanup/duplicate_files/notebook_content_hashes.txt
            fi
        done

        # Group notebooks by content similarity
        if [[ -f .project_cleanup/duplicate_files/notebook_content_hashes.txt ]]; then
            sort .project_cleanup/duplicate_files/notebook_content_hashes.txt | cut -d: -f1 | uniq -c | while read -r count hash; do
                count=$(echo "$count" | xargs)

                if [[ $count -gt 1 ]]; then
                    echo "    📓 Found $count notebooks with similar content (hash: $hash)"
                    grep "^$hash:" .project_cleanup/duplicate_files/notebook_content_hashes.txt >> .project_cleanup/duplicate_files/similar_notebooks.txt
                fi
            done
        fi
    }

    # Execute duplication detection phases
    detect_exact_file_duplicates
    detect_similar_files
    detect_notebook_duplicates

    echo "✅ File duplication detection complete!"
    return 0
}
```

### 3. Comprehensive Unused File Detection System

```bash
# Advanced unused file detection with dependency analysis
detect_unused_files() {
    local target_path="${1:-.}"
    echo "💀 Executing Comprehensive Unused File Detection..."

    # Initialize unused file analysis
    mkdir -p .project_cleanup/unused_files/{
        unreferenced_files,
        orphaned_data,
        unused_configs,
        obsolete_scripts,
        dependency_analysis
    }

    # Python-specific unused file detection
    detect_python_unused_files() {
        echo "  🐍 Analyzing Python project for unused files..."

        # Find all Python files
        find "$target_path" -name "*.py" -not -path "./.project_cleanup/*" | while read -r py_file; do
            echo "$py_file" >> .project_cleanup/unused_files/python_files.txt
        done

        # Find all data files in typical Python project locations
        find "$target_path" -type f \( -path "*/data/*" -o -path "*/datasets/*" -o -path "*/assets/*" \) \
            \( -name "*.csv" -o -name "*.json" -o -name "*.pkl" -o -name "*.npy" -o -name "*.h5" \) \
            -not -path "./.project_cleanup/*" | while read -r data_file; do
            echo "$data_file" >> .project_cleanup/unused_files/python_data_files.txt
        done

        # Check if data files are referenced in Python code
        if [[ -f .project_cleanup/unused_files/python_data_files.txt ]]; then
            while read -r data_file; do
                basename_data=$(basename "$data_file")

                # Check if file is referenced in any Python file
                if ! grep -r -q "$basename_data" "$target_path" --include="*.py" --include="*.ipynb" 2>/dev/null; then
                    echo "$data_file" >> .project_cleanup/unused_files/unreferenced_files/python_unused_data.txt
                    echo "    📊 Potentially unused data file: $data_file"
                fi
            done < .project_cleanup/unused_files/python_data_files.txt
        fi

        # Check for unused Python modules
        if [[ -f .project_cleanup/unused_files/python_files.txt ]]; then
            while read -r py_file; do
                # Skip __init__.py files
                if [[ "$(basename "$py_file")" == "__init__.py" ]]; then
                    continue
                fi

                module_name=$(basename "$py_file" .py)

                # Check if module is imported anywhere
                if ! grep -r -q -E "(import $module_name|from $module_name)" "$target_path" --include="*.py" 2>/dev/null; then
                    # Check if it's a script (has if __name__ == "__main__")
                    if ! grep -q "if __name__ == [\"']__main__[\"']" "$py_file" 2>/dev/null; then
                        echo "$py_file" >> .project_cleanup/unused_files/unreferenced_files/python_unused_modules.txt
                        echo "    🐍 Potentially unused Python module: $py_file"
                    fi
                fi
            done < .project_cleanup/unused_files/python_files.txt
        fi

        # Check for unused Jupyter notebooks
        find "$target_path" -name "*.ipynb" -not -path "./.project_cleanup/*" | while read -r notebook; do
            # Simple heuristic: if notebook hasn't been modified in 6 months and has no recent git activity
            if [[ -f "$notebook" ]]; then
                # Check last modification time (rough heuristic)
                if find "$notebook" -mtime +180 >/dev/null 2>&1; then
                    echo "$notebook" >> .project_cleanup/unused_files/unreferenced_files/old_notebooks.txt
                    echo "    📓 Old notebook (>6 months): $notebook"
                fi
            fi
        done
    }

    # Julia-specific unused file detection
    detect_julia_unused_files() {
        echo "  🟣 Analyzing Julia project for unused files..."

        # Find all Julia files
        find "$target_path" -name "*.jl" -not -path "./.project_cleanup/*" | while read -r jl_file; do
            echo "$jl_file" >> .project_cleanup/unused_files/julia_files.txt
        done

        # Check for unused Julia modules/files
        if [[ -f .project_cleanup/unused_files/julia_files.txt ]]; then
            while read -r jl_file; do
                # Get module name from file
                module_name=$(basename "$jl_file" .jl)

                # Skip special files
                case "$module_name" in
                    "runtests"|"make"|"deps")
                        continue
                        ;;
                esac

                # Check if module is used anywhere
                if ! grep -r -q -E "(using $module_name|import $module_name|include.*$module_name)" "$target_path" --include="*.jl" 2>/dev/null; then
                    echo "$jl_file" >> .project_cleanup/unused_files/unreferenced_files/julia_unused_modules.txt
                    echo "    🟣 Potentially unused Julia module: $jl_file"
                fi
            done < .project_cleanup/unused_files/julia_files.txt
        fi

        # Check for unused data files in Julia projects
        find "$target_path" -type f \( -name "*.csv" -o -name "*.jld2" -o -name "*.arrow" -o -name "*.h5" \) \
            -not -path "./.project_cleanup/*" | while read -r data_file; do
            basename_data=$(basename "$data_file")

            if ! grep -r -q "$basename_data" "$target_path" --include="*.jl" 2>/dev/null; then
                echo "$data_file" >> .project_cleanup/unused_files/unreferenced_files/julia_unused_data.txt
                echo "    📊 Potentially unused Julia data file: $data_file"
            fi
        done

        # Check for obsolete Manifest.toml files
        find "$target_path" -name "Manifest.toml" -not -path "./.project_cleanup/*" | while read -r manifest; do
            manifest_dir=$(dirname "$manifest")

            # Check if there's a corresponding Project.toml
            if [[ ! -f "$manifest_dir/Project.toml" ]]; then
                echo "$manifest" >> .project_cleanup/unused_files/obsolete_scripts/orphaned_manifests.txt
                echo "    🟣 Orphaned Manifest.toml: $manifest"
            fi
        done
    }

    # Generic unused configuration files
    detect_unused_configs() {
        echo "  ⚙️ Detecting unused configuration files..."

        # Common config files that might be obsolete
        local config_patterns=(
            ".coveragerc"
            "tox.ini"
            ".travis.yml"
            "appveyor.yml"
            ".scrutinizer.yml"
            "codecov.yml"
            ".codeclimate.yml"
            "mypy.ini"
            "pytest.ini"
            "setup.cfg"
            ".flake8"
            ".pylintrc"
        )

        for pattern in "${config_patterns[@]}"; do
            find "$target_path" -name "$pattern" -not -path "./.project_cleanup/*" | while read -r config_file; do
                echo "$config_file" >> .project_cleanup/unused_files/found_configs.txt

                # Analyze if config is referenced or used
                case "$pattern" in
                    ".travis.yml"|"appveyor.yml")
                        # Check if CI is actually used
                        echo "$config_file" >> .project_cleanup/unused_files/unused_configs/old_ci_configs.txt
                        echo "    ⚙️ Old CI config found: $config_file"
                        ;;
                    "tox.ini")
                        # Check if tox is in requirements
                        if ! grep -r -q "tox" . --include="*.txt" --include="*.toml" --include="*.cfg" 2>/dev/null; then
                            echo "$config_file" >> .project_cleanup/unused_files/unused_configs/unused_tox.txt
                            echo "    ⚙️ Potentially unused tox config: $config_file"
                        fi
                        ;;
                    ".coveragerc"|"codecov.yml")
                        # Check if coverage is used
                        if ! grep -r -q -E "(coverage|codecov)" . --include="*.txt" --include="*.toml" --include="*.yml" 2>/dev/null; then
                            echo "$config_file" >> .project_cleanup/unused_files/unused_configs/unused_coverage.txt
                            echo "    ⚙️ Potentially unused coverage config: $config_file"
                        fi
                        ;;
                esac
            done
        done
    }

    # Temporary and backup file detection
    detect_temporary_files() {
        echo "  🗑️ Detecting temporary and backup files..."

        # Find common temporary file patterns
        local temp_patterns=(
            "*.tmp"
            "*.temp"
            "*.bak"
            "*.backup"
            "*.old"
            "*~"
            "#*#"
            ".#*"
            "*.swp"
            "*.swo"
            "core.*"
        )

        for pattern in "${temp_patterns[@]}"; do
            find "$target_path" -name "$pattern" -not -path "./.project_cleanup/*" | while read -r temp_file; do
                echo "$temp_file" >> .project_cleanup/unused_files/temporary_files.txt
                echo "    🗑️ Temporary file found: $temp_file"
            done
        done

        # Find old log files
        find "$target_path" -name "*.log" -mtime +30 -not -path "./.project_cleanup/*" | while read -r log_file; do
            echo "$log_file" >> .project_cleanup/unused_files/old_log_files.txt
            echo "    📋 Old log file (>30 days): $log_file"
        done
    }

    # Execute unused file detection phases
    if grep -q "python" .project_cleanup/detected_project_types.txt 2>/dev/null; then
        detect_python_unused_files
    fi

    if grep -q "julia" .project_cleanup/detected_project_types.txt 2>/dev/null; then
        detect_julia_unused_files
    fi

    detect_unused_configs
    detect_temporary_files

    echo "✅ Unused file detection complete!"
    return 0
}
```

### 4. Obsolete Artifact and Cache Detection

```bash
# Comprehensive obsolete artifact detection
detect_obsolete_artifacts() {
    local target_path="${1:-.}"
    echo "🗂️ Executing Obsolete Artifact Detection..."

    # Initialize artifact analysis
    mkdir -p .project_cleanup/obsolete_artifacts/{
        python_artifacts,
        julia_artifacts,
        build_artifacts,
        cache_directories,
        ide_files,
        system_files
    }

    # Python-specific artifact detection
    detect_python_artifacts() {
        echo "  🐍 Detecting Python obsolete artifacts..."

        # __pycache__ directories
        find "$target_path" -type d -name "__pycache__" -not -path "./.project_cleanup/*" | while read -r pycache_dir; do
            echo "$pycache_dir" >> .project_cleanup/obsolete_artifacts/python_artifacts/pycache_dirs.txt

            # Calculate size
            dir_size=$(du -sh "$pycache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo "    🐍 Found __pycache__ directory: $pycache_dir ($dir_size)"
        done

        # .pyc files
        find "$target_path" -name "*.pyc" -not -path "./.project_cleanup/*" | while read -r pyc_file; do
            echo "$pyc_file" >> .project_cleanup/obsolete_artifacts/python_artifacts/pyc_files.txt
            echo "    🐍 Found .pyc file: $pyc_file"
        done

        # .pyo files
        find "$target_path" -name "*.pyo" -not -path "./.project_cleanup/*" | while read -r pyo_file; do
            echo "$pyo_file" >> .project_cleanup/obsolete_artifacts/python_artifacts/pyo_files.txt
            echo "    🐍 Found .pyo file: $pyo_file"
        done

        # Python build directories
        local python_build_dirs=(
            "build"
            "dist"
            "*.egg-info"
            ".eggs"
            ".pytest_cache"
            ".coverage"
            ".tox"
            ".mypy_cache"
            "htmlcov"
        )

        for build_pattern in "${python_build_dirs[@]}"; do
            find "$target_path" -type d -name "$build_pattern" -not -path "./.project_cleanup/*" | while read -r build_dir; do
                echo "$build_dir" >> .project_cleanup/obsolete_artifacts/python_artifacts/build_dirs.txt

                dir_size=$(du -sh "$build_dir" 2>/dev/null | cut -f1 || echo "unknown")
                echo "    🐍 Found Python build directory: $build_dir ($dir_size)"
            done
        done

        # Conda/pip cache (if in project directory)
        find "$target_path" -type d -name ".conda" -o -name "pip-cache" -not -path "./.project_cleanup/*" | while read -r cache_dir; do
            echo "$cache_dir" >> .project_cleanup/obsolete_artifacts/python_artifacts/package_cache.txt

            dir_size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo "    🐍 Found Python package cache: $cache_dir ($dir_size)"
        done
    }

    # Julia-specific artifact detection
    detect_julia_artifacts() {
        echo "  🟣 Detecting Julia obsolete artifacts..."

        # Compiled Julia files (.ji)
        find "$target_path" -name "*.ji" -not -path "./.project_cleanup/*" | while read -r ji_file; do
            echo "$ji_file" >> .project_cleanup/obsolete_artifacts/julia_artifacts/compiled_files.txt
            echo "    🟣 Found compiled Julia file: $ji_file"
        done

        # Julia depot compilation cache (if in project)
        find "$target_path" -type d -path "*/.julia/compiled/*" -not -path "./.project_cleanup/*" | while read -r compiled_dir; do
            echo "$compiled_dir" >> .project_cleanup/obsolete_artifacts/julia_artifacts/compiled_dirs.txt

            dir_size=$(du -sh "$compiled_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo "    🟣 Found Julia compiled directory: $compiled_dir ($dir_size)"
        done

        # Julia package development artifacts
        find "$target_path" -type d -name "deps" | while read -r deps_dir; do
            # Check if it contains build artifacts
            if [[ -f "$deps_dir/build.log" ]] || [[ -f "$deps_dir/deps.jl" ]]; then
                echo "$deps_dir" >> .project_cleanup/obsolete_artifacts/julia_artifacts/deps_dirs.txt

                dir_size=$(du -sh "$deps_dir" 2>/dev/null | cut -f1 || echo "unknown")
                echo "    🟣 Found Julia deps directory: $deps_dir ($dir_size)"
            fi
        done

        # Old Manifest.toml files (if newer exists)
        find "$target_path" -name "Manifest.toml.backup*" -o -name "Manifest.toml.old" -not -path "./.project_cleanup/*" | while read -r old_manifest; do
            echo "$old_manifest" >> .project_cleanup/obsolete_artifacts/julia_artifacts/old_manifests.txt
            echo "    🟣 Found old Manifest backup: $old_manifest"
        done
    }

    # IDE and editor artifacts
    detect_ide_artifacts() {
        echo "  💻 Detecting IDE and editor artifacts..."

        # VS Code
        find "$target_path" -type d -name ".vscode" -not -path "./.project_cleanup/*" | while read -r vscode_dir; do
            echo "$vscode_dir" >> .project_cleanup/obsolete_artifacts/ide_files/vscode_dirs.txt
            echo "    💻 Found VS Code directory: $vscode_dir"
        done

        # JetBrains IDEs
        find "$target_path" -type d -name ".idea" -not -path "./.project_cleanup/*" | while read -r idea_dir; do
            echo "$idea_dir" >> .project_cleanup/obsolete_artifacts/ide_files/idea_dirs.txt
            echo "    💻 Found JetBrains .idea directory: $idea_dir"
        done

        # Vim/Neovim
        find "$target_path" -name ".*.swp" -o -name ".*.swo" -o -name "*~" -not -path "./.project_cleanup/*" | while read -r vim_file; do
            echo "$vim_file" >> .project_cleanup/obsolete_artifacts/ide_files/vim_files.txt
            echo "    💻 Found Vim temporary file: $vim_file"
        done

        # Emacs
        find "$target_path" -name "#*#" -o -name ".#*" -not -path "./.project_cleanup/*" | while read -r emacs_file; do
            echo "$emacs_file" >> .project_cleanup/obsolete_artifacts/ide_files/emacs_files.txt
            echo "    💻 Found Emacs temporary file: $emacs_file"
        done

        # Jupyter
        find "$target_path" -type d -name ".ipynb_checkpoints" -not -path "./.project_cleanup/*" | while read -r checkpoint_dir; do
            echo "$checkpoint_dir" >> .project_cleanup/obsolete_artifacts/ide_files/jupyter_checkpoints.txt

            dir_size=$(du -sh "$checkpoint_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo "    📓 Found Jupyter checkpoint directory: $checkpoint_dir ($dir_size)"
        done
    }

    # System-specific artifacts
    detect_system_artifacts() {
        echo "  🖥️ Detecting system-specific artifacts..."

        # macOS
        find "$target_path" -name ".DS_Store" -not -path "./.project_cleanup/*" | while read -r ds_store; do
            echo "$ds_store" >> .project_cleanup/obsolete_artifacts/system_files/macos_files.txt
            echo "    🍎 Found macOS .DS_Store: $ds_store"
        done

        # Windows
        find "$target_path" -name "Thumbs.db" -o -name "desktop.ini" -not -path "./.project_cleanup/*" | while read -r windows_file; do
            echo "$windows_file" >> .project_cleanup/obsolete_artifacts/system_files/windows_files.txt
            echo "    🪟 Found Windows system file: $windows_file"
        done

        # Linux
        find "$target_path" -name ".directory" -not -path "./.project_cleanup/*" | while read -r linux_file; do
            echo "$linux_file" >> .project_cleanup/obsolete_artifacts/system_files/linux_files.txt
            echo "    🐧 Found Linux .directory file: $linux_file"
        done
    }

    # Execute artifact detection phases
    if grep -q "python" .project_cleanup/detected_project_types.txt 2>/dev/null; then
        detect_python_artifacts
    fi

    if grep -q "julia" .project_cleanup/detected_project_types.txt 2>/dev/null; then
        detect_julia_artifacts
    fi

    detect_ide_artifacts
    detect_system_artifacts

    echo "✅ Obsolete artifact detection complete!"
    return 0
}
```

### 5. Empty Directory Detection

```bash
# Comprehensive empty directory detection
detect_empty_directories() {
    local target_path="${1:-.}"
    echo "📁 Detecting Empty Directories..."

    # Initialize empty directory analysis
    mkdir -p .project_cleanup/empty_directories

    # Find truly empty directories
    find "$target_path" -type d -empty -not -path "./.project_cleanup/*" -not -path "./.git/*" | while read -r empty_dir; do
        echo "$empty_dir" >> .project_cleanup/empty_directories/truly_empty.txt
        echo "  📁 Truly empty directory: $empty_dir"
    done

    # Find directories with only hidden files
    find "$target_path" -type d -not -path "./.project_cleanup/*" -not -path "./.git/*" | while read -r dir; do
        # Count visible files
        visible_count=$(find "$dir" -maxdepth 1 -not -name ".*" -not -name "$(basename "$dir")" | wc -l)

        # Count all files
        total_count=$(find "$dir" -maxdepth 1 -not -name "$(basename "$dir")" | wc -l)

        # If directory has files but all are hidden
        if [[ $visible_count -eq 0 && $total_count -gt 0 ]]; then
            echo "$dir" >> .project_cleanup/empty_directories/only_hidden_files.txt
            echo "  👻 Directory with only hidden files: $dir"
        fi
    done

    # Find directories with only system junk
    find "$target_path" -type d -not -path "./.project_cleanup/*" -not -path "./.git/*" | while read -r dir; do
        # Check if directory contains only system files
        file_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
        junk_count=$(find "$dir" -maxdepth 1 -type f \( -name ".DS_Store" -o -name "Thumbs.db" -o -name "desktop.ini" \) | wc -l)

        if [[ $file_count -gt 0 && $file_count -eq $junk_count ]]; then
            echo "$dir" >> .project_cleanup/empty_directories/only_system_junk.txt
            echo "  🗑️ Directory with only system junk: $dir"
        fi
    done

    echo "✅ Empty directory detection complete!"
    return 0
}
```

### 6. Git Integration and Analysis

```bash
# Comprehensive git integration for cleanup analysis
analyze_git_status() {
    local target_path="${1:-.}"
    echo "🔀 Analyzing Git Repository Status..."

    # Check if this is a git repository
    if ! git -C "$target_path" rev-parse --git-dir >/dev/null 2>&1; then
        echo "  ⚠️ Not a git repository - skipping git analysis"
        return 0
    fi

    # Initialize git analysis
    mkdir -p .project_cleanup/git_analysis/{
        untracked_files,
        ignored_files,
        large_files,
        old_branches,
        commit_analysis
    }

    # Analyze untracked files
    analyze_untracked_files() {
        echo "  📊 Analyzing untracked files..."

        git -C "$target_path" ls-files --others --exclude-standard > .project_cleanup/git_analysis/untracked_files/untracked.txt

        if [[ -s .project_cleanup/git_analysis/untracked_files/untracked.txt ]]; then
            echo "    📋 Found untracked files:"
            while read -r untracked_file; do
                echo "      📄 $untracked_file"

                # Categorize untracked files
                case "$untracked_file" in
                    *.pyc|*.pyo|__pycache__/*)
                        echo "$untracked_file" >> .project_cleanup/git_analysis/untracked_files/python_artifacts.txt
                        ;;
                    *.ji|*.so|*.dylib)
                        echo "$untracked_file" >> .project_cleanup/git_analysis/untracked_files/compiled_artifacts.txt
                        ;;
                    *.log|*.tmp|*.temp)
                        echo "$untracked_file" >> .project_cleanup/git_analysis/untracked_files/temp_files.txt
                        ;;
                    .DS_Store|Thumbs.db)
                        echo "$untracked_file" >> .project_cleanup/git_analysis/untracked_files/system_junk.txt
                        ;;
                    *)
                        echo "$untracked_file" >> .project_cleanup/git_analysis/untracked_files/other_files.txt
                        ;;
                esac
            done < .project_cleanup/git_analysis/untracked_files/untracked.txt
        else
            echo "    ✅ No untracked files found"
        fi
    }

    # Analyze files that should be in .gitignore
    analyze_gitignore_violations() {
        echo "  🚫 Analyzing potential .gitignore violations..."

        # Check if tracked files match common ignore patterns
        local ignore_patterns=(
            "*.pyc" "*.pyo" "__pycache__/"
            "*.ji" "*.so" "*.dylib"
            ".DS_Store" "Thumbs.db"
            "*.log" "*.tmp"
            ".env" ".env.local"
            "node_modules/" ".npm/"
            ".vscode/" ".idea/"
        )

        for pattern in "${ignore_patterns[@]}"; do
            # Convert glob pattern to find expression
            case "$pattern" in
                "*.*")
                    extension="${pattern#*.}"
                    git -C "$target_path" ls-files | grep "\.$extension$" >> .project_cleanup/git_analysis/ignored_files/should_be_ignored.txt 2>/dev/null || true
                    ;;
                "*/")
                    dirname="${pattern%/}"
                    git -C "$target_path" ls-files | grep "^$dirname/" >> .project_cleanup/git_analysis/ignored_files/should_be_ignored.txt 2>/dev/null || true
                    ;;
                *)
                    git -C "$target_path" ls-files | grep "$pattern" >> .project_cleanup/git_analysis/ignored_files/should_be_ignored.txt 2>/dev/null || true
                    ;;
            esac
        done

        if [[ -s .project_cleanup/git_analysis/ignored_files/should_be_ignored.txt ]]; then
            echo "    ⚠️ Found tracked files that should probably be ignored:"
            sort .project_cleanup/git_analysis/ignored_files/should_be_ignored.txt | uniq | while read -r ignored_file; do
                echo "      🚫 $ignored_file"
            done
        fi
    }

    # Identify large files that should use Git LFS
    analyze_large_files() {
        echo "  📏 Analyzing large files for Git LFS candidates..."

        # Find files larger than 50MB
        git -C "$target_path" ls-files | while read -r tracked_file; do
            if [[ -f "$target_path/$tracked_file" ]]; then
                file_size=$(stat -f%z "$target_path/$tracked_file" 2>/dev/null || stat -c%s "$target_path/$tracked_file" 2>/dev/null || echo "0")

                # Files larger than 50MB
                if [[ $file_size -gt 52428800 ]]; then
                    echo "$tracked_file:$file_size" >> .project_cleanup/git_analysis/large_files/lfs_candidates.txt

                    # Convert size to human readable
                    human_size=$(du -h "$target_path/$tracked_file" 2>/dev/null | cut -f1 || echo "unknown")
                    echo "    📏 Large file (LFS candidate): $tracked_file ($human_size)"
                fi
            fi
        done
    }

    # Analyze old branches
    analyze_old_branches() {
        echo "  🌿 Analyzing old and merged branches..."

        # List all branches with last commit date
        git -C "$target_path" for-each-ref --format='%(refname:short) %(committerdate:iso8601) %(upstream:track)' refs/heads/ > .project_cleanup/git_analysis/old_branches/all_branches.txt

        # Find branches older than 3 months
        local three_months_ago=$(date -d '3 months ago' '+%Y-%m-%d' 2>/dev/null || date -v-3m '+%Y-%m-%d' 2>/dev/null || echo "2024-01-01")

        while read -r branch_info; do
            branch_name=$(echo "$branch_info" | cut -d' ' -f1)
            commit_date=$(echo "$branch_info" | cut -d' ' -f2)

            if [[ "$commit_date" < "$three_months_ago" ]]; then
                echo "$branch_info" >> .project_cleanup/git_analysis/old_branches/old_branches.txt
                echo "    🌿 Old branch: $branch_name (last commit: $commit_date)"
            fi
        done < .project_cleanup/git_analysis/old_branches/all_branches.txt 2>/dev/null || true

        # Find merged branches
        git -C "$target_path" branch --merged main 2>/dev/null | grep -v "main\|master" > .project_cleanup/git_analysis/old_branches/merged_branches.txt || true
        git -C "$target_path" branch --merged master 2>/dev/null | grep -v "main\|master" >> .project_cleanup/git_analysis/old_branches/merged_branches.txt || true

        if [[ -s .project_cleanup/git_analysis/old_branches/merged_branches.txt ]]; then
            echo "    🌿 Found merged branches:"
            while read -r merged_branch; do
                merged_branch=$(echo "$merged_branch" | xargs)  # trim whitespace
                echo "      🔀 $merged_branch"
            done < .project_cleanup/git_analysis/old_branches/merged_branches.txt
        fi
    }

    # Execute git analysis phases
    analyze_untracked_files
    analyze_gitignore_violations
    analyze_large_files
    analyze_old_branches

    echo "✅ Git analysis complete!"
    return 0
}
```

### 7. Project Cleanup Plan Generation and Execution

```bash
# Generate comprehensive project cleanup plan
generate_project_cleanup_plan() {
    echo "📋 Generating Comprehensive Project Cleanup Plan..."

    # Initialize project cleanup plan structure
    mkdir -p .project_cleanup/cleanup_plan/{high_priority,medium_priority,low_priority,reports,execution_scripts}

    # Calculate cleanup impact and metrics
    calculate_project_metrics() {
        echo "  📊 Calculating project cleanup metrics..."

        local total_duplicate_files=0
        local total_unused_files=0
        local total_obsolete_artifacts=0
        local total_empty_directories=0
        local total_untracked_files=0
        local estimated_space_savings=0

        # Count duplicate files
        if [[ -f .project_cleanup/duplicate_files/exact_duplicates_summary.txt ]]; then
            total_duplicate_files=$(grep -c "📍" .project_cleanup/duplicate_files/exact_duplicates_summary.txt 2>/dev/null || echo "0")
        fi

        # Count unused files
        for unused_category in .project_cleanup/unused_files/unreferenced_files/*.txt; do
            if [[ -f "$unused_category" ]]; then
                category_count=$(wc -l < "$unused_category" 2>/dev/null || echo "0")
                total_unused_files=$((total_unused_files + category_count))
            fi
        done

        # Count obsolete artifacts
        for artifact_category in .project_cleanup/obsolete_artifacts/*/*.txt; do
            if [[ -f "$artifact_category" ]]; then
                artifact_count=$(wc -l < "$artifact_category" 2>/dev/null || echo "0")
                total_obsolete_artifacts=$((total_obsolete_artifacts + artifact_count))
            fi
        done

        # Count empty directories
        if [[ -f .project_cleanup/empty_directories/truly_empty.txt ]]; then
            total_empty_directories=$(wc -l < .project_cleanup/empty_directories/truly_empty.txt 2>/dev/null || echo "0")
        fi

        # Count untracked files
        if [[ -f .project_cleanup/git_analysis/untracked_files/untracked.txt ]]; then
            total_untracked_files=$(wc -l < .project_cleanup/git_analysis/untracked_files/untracked.txt 2>/dev/null || echo "0")
        fi

        # Store metrics
        cat > .project_cleanup/cleanup_plan/metrics.json << EOF
{
    "total_duplicate_files": $total_duplicate_files,
    "total_unused_files": $total_unused_files,
    "total_obsolete_artifacts": $total_obsolete_artifacts,
    "total_empty_directories": $total_empty_directories,
    "total_untracked_files": $total_untracked_files,
    "estimated_space_savings_mb": $((total_obsolete_artifacts * 10 + total_duplicate_files * 5)),
    "complexity_score": $((total_duplicate_files * 2 + total_unused_files + total_obsolete_artifacts * 3 + total_empty_directories))
}
EOF

        echo "    📈 Project cleanup metrics calculated:"
        echo "      🔄 Duplicate files: $total_duplicate_files"
        echo "      💀 Unused files: $total_unused_files"
        echo "      🗂️ Obsolete artifacts: $total_obsolete_artifacts"
        echo "      📁 Empty directories: $total_empty_directories"
        echo "      📊 Untracked files: $total_untracked_files"
    }

    # Generate prioritized cleanup tasks
    generate_project_cleanup_tasks() {
        echo "  🎯 Generating prioritized project cleanup tasks..."

        # High Priority Tasks (Safe, High Impact)
        cat > .project_cleanup/cleanup_plan/high_priority/tasks.md << 'EOF'
# High Priority Project Cleanup Tasks

## 1. Remove Build Artifacts and Cache Files (Risk: Very Low, Impact: High)
- **Description**: Remove Python __pycache__, .pyc files, Julia compiled files, and build directories
- **Benefits**: Immediate space savings, cleaner repository, faster git operations
- **Risk Assessment**: Very Low - These files are regenerated automatically
- **Automation**: Fully automated
- **Estimated Space Savings**: 50-200MB typically

## 2. Clean System Junk Files (Risk: Very Low, Impact: Medium)
- **Description**: Remove .DS_Store, Thumbs.db, and other OS-generated files
- **Benefits**: Cleaner repository, better cross-platform compatibility
- **Risk Assessment**: Very Low - OS will regenerate as needed
- **Automation**: Fully automated
- **Estimated Space Savings**: 1-10MB

## 3. Remove Empty Directories (Risk: Very Low, Impact: Medium)
- **Description**: Remove truly empty directories and directories with only system junk
- **Benefits**: Cleaner project structure, reduced directory bloat
- **Risk Assessment**: Very Low - Can be easily recreated if needed
- **Automation**: Fully automated
- **Estimated Space Savings**: Minimal, but improves structure

## 4. Clean IDE and Editor Temporary Files (Risk: Very Low, Impact: Medium)
- **Description**: Remove .swp, .swo, #*#, .ipynb_checkpoints, etc.
- **Benefits**: Cleaner repository, no editor conflicts
- **Risk Assessment**: Very Low - Temporary files, safe to remove
- **Automation**: Fully automated
- **Estimated Space Savings**: 5-50MB
EOF

        # Medium Priority Tasks (Moderate Risk/Impact)
        cat > .project_cleanup/cleanup_plan/medium_priority/tasks.md << 'EOF'
# Medium Priority Project Cleanup Tasks

## 1. Remove Exact Duplicate Files (Risk: Low, Impact: High)
- **Description**: Remove files with identical content, keeping one copy
- **Benefits**: Significant space savings, eliminates redundancy
- **Risk Assessment**: Low - Content identical, safe to remove duplicates
- **Automation**: Semi-automated with user confirmation for file selection
- **Estimated Space Savings**: Variable, can be substantial

## 2. Update .gitignore for Tracked Artifacts (Risk: Low, Impact: High)
- **Description**: Add build artifacts and system files to .gitignore, remove from tracking
- **Benefits**: Cleaner git history, smaller repository size
- **Risk Assessment**: Low - Using git rm --cached preserves local files
- **Automation**: Semi-automated with .gitignore updates
- **Estimated Space Savings**: Repository size reduction

## 3. Clean Old Log and Temporary Files (Risk: Low, Impact: Medium)
- **Description**: Remove old log files and temporary files older than 30 days
- **Benefits**: Space savings, cleaner project structure
- **Risk Assessment**: Low - Old files, usually not needed
- **Automation**: Semi-automated with age-based filtering
- **Estimated Space Savings**: 10-100MB

## 4. Archive or Remove Old Configuration Files (Risk: Medium, Impact: Medium)
- **Description**: Remove obsolete CI configs, unused tool configurations
- **Benefits**: Reduced maintenance overhead, cleaner project
- **Risk Assessment**: Medium - Some configs might still be used
- **Automation**: Manual review recommended
- **Estimated Space Savings**: 1-10MB
EOF

        # Low Priority Tasks (Variable Risk/Impact)
        cat > .project_cleanup/cleanup_plan/low_priority/tasks.md << 'EOF'
# Low Priority Project Cleanup Tasks

## 1. Review Potentially Unused Data Files (Risk: High, Impact: Variable)
- **Description**: Review data files that aren't obviously referenced in code
- **Benefits**: Space savings, reduced data clutter
- **Risk Assessment**: High - Data files might be used in non-obvious ways
- **Automation**: Manual review required
- **Estimated Space Savings**: Highly variable

## 2. Consolidate Similar Files (Risk: High, Impact: Medium)
- **Description**: Review files with same name in different locations
- **Benefits**: Reduced confusion, better organization
- **Risk Assessment**: High - Files might serve different purposes
- **Automation**: Manual analysis and decision required
- **Estimated Space Savings**: Variable

## 3. Clean Old Git Branches (Risk: Medium, Impact: Low)
- **Description**: Remove old merged branches and branches inactive for >3 months
- **Benefits**: Cleaner git history, easier navigation
- **Risk Assessment**: Medium - Branches might contain important history
- **Automation**: Semi-automated with confirmation
- **Estimated Space Savings**: Minimal

## 4. Archive Old Jupyter Notebooks (Risk: Medium, Impact: Low)
- **Description**: Move or archive notebooks not modified in 6+ months
- **Benefits**: Cleaner project structure, focus on active work
- **Risk Assessment**: Medium - Notebooks might contain important analysis
- **Automation**: Manual review required
- **Estimated Space Savings**: Variable
EOF
    }

    # Generate execution scripts for automated tasks
    generate_execution_scripts() {
        echo "  📝 Generating execution scripts..."

        # High priority automated cleanup script
        cat > .project_cleanup/cleanup_plan/execution_scripts/high_priority_cleanup.sh << 'EOF'
#!/bin/bash
# High Priority Automated Project Cleanup
# This script performs safe, automated cleanup tasks

set -euo pipefail

echo "🚀 Starting High Priority Project Cleanup..."

# Remove Python cache files
echo "🐍 Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove Julia compiled files
echo "🟣 Cleaning Julia compiled files..."
find . -name "*.ji" -delete 2>/dev/null || true

# Remove system junk
echo "🗑️ Cleaning system junk files..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true
find . -name "desktop.ini" -delete 2>/dev/null || true

# Remove editor temporary files
echo "💻 Cleaning editor temporary files..."
find . -name "*.swp" -delete 2>/dev/null || true
find . -name "*.swo" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true
find . -name "#*#" -delete 2>/dev/null || true
find . -name ".#*" -delete 2>/dev/null || true

# Remove Jupyter checkpoints
echo "📓 Cleaning Jupyter checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# Remove empty directories
echo "📁 Cleaning empty directories..."
find . -type d -empty -not -path "./.git/*" -not -path "./.project_cleanup/*" -delete 2>/dev/null || true

echo "✅ High Priority Cleanup Complete!"
EOF

        chmod +x .project_cleanup/cleanup_plan/execution_scripts/high_priority_cleanup.sh

        # Medium priority script (requires confirmation)
        cat > .project_cleanup/cleanup_plan/execution_scripts/medium_priority_cleanup.sh << 'EOF'
#!/bin/bash
# Medium Priority Project Cleanup (Requires Confirmation)
# This script performs cleanup tasks that require user confirmation

set -euo pipefail

echo "🚀 Starting Medium Priority Project Cleanup..."

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Clean old log files
if [[ -f .project_cleanup/unused_files/old_log_files.txt ]]; then
    echo "📋 Found old log files:"
    cat .project_cleanup/unused_files/old_log_files.txt
    if confirm "Remove these old log files?"; then
        while read -r log_file; do
            rm -f "$log_file"
            echo "  🗑️ Removed: $log_file"
        done < .project_cleanup/unused_files/old_log_files.txt
    fi
fi

# Update .gitignore for common artifacts
if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "🚫 Updating .gitignore for common artifacts..."

    # Add common patterns to .gitignore if not already present
    patterns=(
        "__pycache__/"
        "*.pyc"
        "*.pyo"
        "*.ji"
        ".DS_Store"
        "Thumbs.db"
        ".ipynb_checkpoints/"
        "*.swp"
        "*.swo"
        "*~"
    )

    for pattern in "${patterns[@]}"; do
        if ! grep -q "^$pattern" .gitignore 2>/dev/null; then
            echo "$pattern" >> .gitignore
            echo "  ✅ Added to .gitignore: $pattern"
        fi
    done
fi

echo "✅ Medium Priority Cleanup Complete!"
EOF

        chmod +x .project_cleanup/cleanup_plan/execution_scripts/medium_priority_cleanup.sh
    }

    # Execute plan generation
    calculate_project_metrics
    generate_project_cleanup_tasks
    generate_execution_scripts

    echo "✅ Project cleanup plan generation complete!"
    return 0
}

# Interactive confirmation and execution system
interactive_project_confirmation() {
    echo "🤝 Initiating Interactive Project Cleanup Confirmation..."

    # Display project cleanup summary
    display_project_summary() {
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════════════╗"
        echo "║                         PROJECT CLEANUP SUMMARY                               ║"
        echo "╚════════════════════════════════════════════════════════════════════════════════╝"
        echo ""

        # Read metrics
        if [[ -f .project_cleanup/cleanup_plan/metrics.json ]]; then
            local duplicate_files=$(jq -r '.total_duplicate_files' .project_cleanup/cleanup_plan/metrics.json 2>/dev/null || echo "0")
            local unused_files=$(jq -r '.total_unused_files' .project_cleanup/cleanup_plan/metrics.json 2>/dev/null || echo "0")
            local obsolete_artifacts=$(jq -r '.total_obsolete_artifacts' .project_cleanup/cleanup_plan/metrics.json 2>/dev/null || echo "0")
            local empty_directories=$(jq -r '.total_empty_directories' .project_cleanup/cleanup_plan/metrics.json 2>/dev/null || echo "0")
            local untracked_files=$(jq -r '.total_untracked_files' .project_cleanup/cleanup_plan/metrics.json 2>/dev/null || echo "0")
            local space_savings=$(jq -r '.estimated_space_savings_mb' .project_cleanup/cleanup_plan/metrics.json 2>/dev/null || echo "0")

            echo "📊 PROJECT CLEANUP OPPORTUNITIES:"
            echo ""
            echo "  🔄 Duplicate Files:        $duplicate_files"
            echo "  💀 Unused Files:           $unused_files"
            echo "  🗂️ Obsolete Artifacts:     $obsolete_artifacts"
            echo "  📁 Empty Directories:      $empty_directories"
            echo "  📊 Untracked Files:        $untracked_files"
            echo ""
            echo "  💾 Estimated Space Savings: ${space_savings}MB"
            echo ""
        fi

        # Project type information
        if [[ -f .project_cleanup/detected_project_types.txt ]]; then
            echo "🎯 DETECTED PROJECT TYPES:"
            echo ""
            while read -r project_type; do
                case "$project_type" in
                    python) echo "  🐍 Python Scientific Computing Project" ;;
                    julia) echo "  🟣 Julia Scientific Computing Project" ;;
                    mixed) echo "  🔄 Mixed Python/Julia Project" ;;
                esac
            done < .project_cleanup/detected_project_types.txt
            echo ""
        fi

        echo "🎯 CLEANUP PHASES AVAILABLE:"
        echo ""
        echo "  1️⃣  High Priority (Safe, Automated)     - Cache files, system junk, empty dirs"
        echo "  2️⃣  Medium Priority (Semi-Automated)    - Duplicates, git cleanup, configs"
        echo "  3️⃣  Low Priority (Manual Review)        - Data files, notebooks, branches"
        echo ""
        echo "🛡️ SAFETY FEATURES:"
        echo "  ✅ Automatic backup creation"
        echo "  ✅ Git integration with stash capability"
        echo "  ✅ Dry-run mode available"
        echo "  ✅ Incremental execution with validation"
        echo ""
    }

    # Get user confirmation
    get_project_confirmation() {
        echo "❓ PROJECT CLEANUP OPTIONS:"
        echo ""
        echo "  [1] Execute high-priority cleanup only (recommended, safe)"
        echo "  [2] Execute high + medium priority cleanup"
        echo "  [3] Execute all cleanup phases (full project cleanup)"
        echo "  [4] Custom selection (choose specific cleanup categories)"
        echo "  [5] Dry-run only (preview changes without executing)"
        echo "  [6] Generate report and exit (no cleanup)"
        echo "  [7] Run automated scripts only"
        echo "  [0] Cancel cleanup"
        echo ""

        while true; do
            read -p "🔍 Please select an option [0-7]: " choice

            case $choice in
                1)
                    echo "✅ Selected: High-priority project cleanup (safe automated)"
                    return 1
                    ;;
                2)
                    echo "✅ Selected: High + Medium priority project cleanup"
                    return 2
                    ;;
                3)
                    echo "✅ Selected: Full project cleanup (all phases)"
                    return 3
                    ;;
                4)
                    echo "✅ Selected: Custom cleanup selection"
                    return 4
                    ;;
                5)
                    echo "✅ Selected: Dry-run mode"
                    return 5
                    ;;
                6)
                    echo "✅ Selected: Report generation only"
                    return 6
                    ;;
                7)
                    echo "✅ Selected: Run automated scripts"
                    return 7
                    ;;
                0)
                    echo "❌ Project cleanup cancelled by user"
                    return 0
                    ;;
                *)
                    echo "❌ Invalid option. Please select 0-7."
                    ;;
            esac
        done
    }

    # Execute based on user choice
    display_project_summary
    get_project_confirmation
    local user_choice=$?

    return $user_choice
}

# Main execution controller
clean_project() {
    local target_path="${1:-.}"
    local dry_run="${dry_run:-false}"
    local aggressive="${aggressive:-false}"
    local project_type="${project_type:-auto}"
    local interactive="${interactive:-true}"
    local backup="${backup:-true}"
    local git_integration="${git_integration:-true}"
    local report="${report:-true}"

    echo "🗂️ Revolutionary Project Cleanup Engine Starting..."
    echo "🎯 Target: $target_path"
    echo "🔧 Mode: $([ "$dry_run" == "true" ] && echo "Dry Run" || echo "Live Execution")"
    echo "📦 Type: $project_type"
    echo ""

    # Validate target path
    if [[ ! -d "$target_path" ]]; then
        echo "❌ Error: Target path '$target_path' does not exist"
        return 1
    fi

    # Initialize cleanup environment
    cd "$target_path" || return 1

    # Execute analysis phases
    analyze_project_ecosystem "$target_path"
    detect_file_duplicates "$target_path"
    detect_unused_files "$target_path"
    detect_obsolete_artifacts "$target_path"
    detect_empty_directories "$target_path"

    if [[ "$git_integration" == "true" ]]; then
        analyze_git_status "$target_path"
    fi

    generate_project_cleanup_plan

    # Interactive confirmation (unless disabled)
    if [[ "$interactive" == "true" ]]; then
        interactive_project_confirmation
        local user_choice=$?

        case $user_choice in
            0)
                # User cancelled
                if [[ "$report" == "true" ]]; then
                    generate_project_report
                fi
                return 0
                ;;
            6)
                # Report only
                generate_project_report
                return 0
                ;;
            5)
                # Dry run
                echo "🔍 Dry-run mode: Analysis complete, no changes made"
                generate_project_report
                return 0
                ;;
            7)
                # Run automated scripts
                if [[ "$dry_run" != "true" ]]; then
                    echo "🚀 Executing automated cleanup scripts..."
                    bash .project_cleanup/cleanup_plan/execution_scripts/high_priority_cleanup.sh
                fi
                ;;
            *)
                # Execute cleanup
                if [[ "$dry_run" != "true" ]]; then
                    execute_project_cleanup "$user_choice"
                fi
                ;;
        esac
    fi

    # Generate final report
    if [[ "$report" == "true" ]]; then
        generate_project_report
    fi

    echo ""
    echo "🎉 Project cleanup analysis complete!"
    echo "📊 Review the generated report and analysis files"
    echo "💡 Use automated scripts in .project_cleanup/cleanup_plan/execution_scripts/"

    return 0
}

# Generate comprehensive project report
generate_project_report() {
    echo "📊 Generating Comprehensive Project Analysis Report..."

    local report_file=".project_cleanup/reports/project_cleanup_report_$(date +%Y%m%d_%H%M%S).md"
    local summary_file=".project_cleanup/reports/cleanup_summary.txt"

    # Ensure reports directory exists
    mkdir -p .project_cleanup/reports

    # Generate detailed markdown report
    cat > "$report_file" << 'EOF'
# 🗂️ Project Cleanup Analysis Report

## Executive Summary
EOF

    # Add project type detection results
    if [[ -f .project_cleanup/detected_project_types.txt ]]; then
        echo "" >> "$report_file"
        echo "### 🏗️ Project Type Analysis" >> "$report_file"
        echo "**Detected Project Types:**" >> "$report_file"
        while read -r project_type; do
            echo "- $project_type" >> "$report_file"
        done < .project_cleanup/detected_project_types.txt
    fi

    # Add confidence scores
    if [[ -f .project_cleanup/confidence_scores.txt ]]; then
        echo "" >> "$report_file"
        echo "**Confidence Scores:**" >> "$report_file"
        echo '```' >> "$report_file"
        cat .project_cleanup/confidence_scores.txt >> "$report_file"
        echo '```' >> "$report_file"
    fi

    # File inventory summary
    echo "" >> "$report_file"
    echo "### 📁 File Inventory Analysis" >> "$report_file"

    local total_files=0
    local total_size=0

    if [[ -d .project_cleanup/file_inventory ]]; then
        # Count files by category
        for category in .project_cleanup/file_inventory/*/files.txt; do
            if [[ -f "$category" ]]; then
                category_name=$(basename $(dirname "$category"))
                file_count=$(wc -l < "$category")
                total_files=$((total_files + file_count))
                echo "- **$category_name:** $file_count files" >> "$report_file"
            fi
        done

        # Calculate total project size
        if [[ -f .project_cleanup/file_inventory/summary.txt ]]; then
            total_size=$(grep "Total project size" .project_cleanup/file_inventory/summary.txt 2>/dev/null | awk '{print $4}' || echo "Unknown")
        fi
    fi

    echo "" >> "$report_file"
    echo "**Total Files:** $total_files" >> "$report_file"
    echo "**Total Size:** $total_size" >> "$report_file"

    # Duplicate files analysis
    echo "" >> "$report_file"
    echo "### 🔄 Duplicate Files Analysis" >> "$report_file"

    if [[ -f .project_cleanup/duplicate_files/exact_duplicates.txt ]]; then
        local duplicate_count=$(wc -l < .project_cleanup/duplicate_files/exact_duplicates.txt)
        echo "- **Exact Duplicates Found:** $duplicate_count" >> "$report_file"

        if [[ $duplicate_count -gt 0 ]]; then
            echo "" >> "$report_file"
            echo "**Duplicate File Groups:**" >> "$report_file"
            echo '```' >> "$report_file"
            head -20 .project_cleanup/duplicate_files/exact_duplicates.txt >> "$report_file"
            if [[ $duplicate_count -gt 20 ]]; then
                echo "... ($(($duplicate_count - 20)) more entries)" >> "$report_file"
            fi
            echo '```' >> "$report_file"
        fi
    else
        echo "- **Exact Duplicates Found:** 0" >> "$report_file"
    fi

    # Unused files analysis
    echo "" >> "$report_file"
    echo "### 🗑️ Unused Files Analysis" >> "$report_file"

    if [[ -f .project_cleanup/unused_files/unused_files.txt ]]; then
        local unused_count=$(wc -l < .project_cleanup/unused_files/unused_files.txt)
        echo "- **Unused Files Found:** $unused_count" >> "$report_file"

        if [[ $unused_count -gt 0 ]]; then
            echo "" >> "$report_file"
            echo "**Sample Unused Files:**" >> "$report_file"
            echo '```' >> "$report_file"
            head -10 .project_cleanup/unused_files/unused_files.txt >> "$report_file"
            if [[ $unused_count -gt 10 ]]; then
                echo "... ($(($unused_count - 10)) more files)" >> "$report_file"
            fi
            echo '```' >> "$report_file"
        fi
    else
        echo "- **Unused Files Found:** 0" >> "$report_file"
    fi

    # Empty directories
    echo "" >> "$report_file"
    echo "### 📂 Empty Directories" >> "$report_file"

    if [[ -f .project_cleanup/empty_directories/empty_dirs.txt ]]; then
        local empty_count=$(wc -l < .project_cleanup/empty_directories/empty_dirs.txt)
        echo "- **Empty Directories Found:** $empty_count" >> "$report_file"

        if [[ $empty_count -gt 0 ]]; then
            echo "" >> "$report_file"
            echo "**Empty Directories:**" >> "$report_file"
            echo '```' >> "$report_file"
            cat .project_cleanup/empty_directories/empty_dirs.txt >> "$report_file"
            echo '```' >> "$report_file"
        fi
    else
        echo "- **Empty Directories Found:** 0" >> "$report_file"
    fi

    # Cleanup recommendations
    echo "" >> "$report_file"
    echo "### 💡 Cleanup Recommendations" >> "$report_file"

    if [[ -f .project_cleanup/cleanup_plans/recommendations.txt ]]; then
        echo "" >> "$report_file"
        cat .project_cleanup/cleanup_plans/recommendations.txt >> "$report_file"
    else
        echo "" >> "$report_file"
        echo "1. **Review Duplicate Files:** Consider removing or consolidating identical files" >> "$report_file"
        echo "2. **Clean Unused Files:** Archive or remove files not referenced in the project" >> "$report_file"
        echo "3. **Remove Empty Directories:** Clean up directory structure" >> "$report_file"
        echo "4. **Optimize Dependencies:** Review and clean up package dependencies" >> "$report_file"
    fi

    # Add scientific computing specific recommendations
    if [[ -f .project_cleanup/scientific_indicators.txt ]]; then
        echo "" >> "$report_file"
        echo "### 🔬 Scientific Computing Recommendations" >> "$report_file"
        echo "- **Environment Management:** Consider using conda/mamba for Python or Pkg for Julia" >> "$report_file"
        echo "- **Reproducibility:** Ensure all dependencies are pinned with version constraints" >> "$report_file"
        echo "- **Data Management:** Organize datasets and results with clear naming conventions" >> "$report_file"
        echo "- **Documentation:** Include methodology and computational environment details" >> "$report_file"
    fi

    # Git integration results
    if [[ -d .project_cleanup/git_analysis ]]; then
        echo "" >> "$report_file"
        echo "### 🔀 Git Repository Analysis" >> "$report_file"

        if [[ -f .project_cleanup/git_analysis/untracked_files.txt ]]; then
            local untracked_count=$(wc -l < .project_cleanup/git_analysis/untracked_files.txt)
            echo "- **Untracked Files:** $untracked_count" >> "$report_file"
        fi

        if [[ -f .project_cleanup/git_analysis/ignored_files.txt ]]; then
            local ignored_count=$(wc -l < .project_cleanup/git_analysis/ignored_files.txt)
            echo "- **Ignored Files:** $ignored_count" >> "$report_file"
        fi
    fi

    # Add footer
    echo "" >> "$report_file"
    echo "---" >> "$report_file"
    echo "*Report generated on $(date) by Revolutionary Project Cleanup Engine*" >> "$report_file"
    echo "" >> "$report_file"
    echo "**Next Steps:**" >> "$report_file"
    echo "1. Review the analysis results and recommendations" >> "$report_file"
    echo "2. Use the generated cleanup scripts in .project_cleanup/cleanup_plans/execution_scripts/" >> "$report_file"
    echo "3. Create backups before executing any cleanup operations" >> "$report_file"
    echo "4. Test your project after cleanup to ensure functionality" >> "$report_file"

    # Generate simple text summary
    cat > "$summary_file" << EOF
PROJECT CLEANUP ANALYSIS SUMMARY
================================
Generated: $(date)
Target: $(pwd)

Files Analyzed: $total_files
Project Size: $total_size

Issues Found:
- Duplicate files: $(test -f .project_cleanup/duplicate_files/exact_duplicates.txt && wc -l < .project_cleanup/duplicate_files/exact_duplicates.txt || echo "0")
- Unused files: $(test -f .project_cleanup/unused_files/unused_files.txt && wc -l < .project_cleanup/unused_files/unused_files.txt || echo "0")
- Empty directories: $(test -f .project_cleanup/empty_directories/empty_dirs.txt && wc -l < .project_cleanup/empty_directories/empty_dirs.txt || echo "0")

Detailed report: $report_file
EOF

    echo "✅ Project analysis report generated:"
    echo "  📋 Detailed report: $report_file"
    echo "  📄 Summary: $summary_file"
    echo ""
    echo "🔍 Key Findings:"

    # Display summary findings
    if [[ -f "$summary_file" ]]; then
        grep -E "(Duplicate files|Unused files|Empty directories):" "$summary_file" | while read -r finding; do
            echo "  • $finding"
        done
    fi

    echo ""
    echo "💡 Review the detailed report for comprehensive analysis and recommendations."
}

# Show usage information
show_usage() {
    cat << 'EOF'
🗂️ Revolutionary Project Cleanup Engine

USAGE:
    clean-project [target-path] [options]

ARGUMENTS:
    target-path     Directory to analyze (default: current directory)

OPTIONS:
    --dry-run              Preview changes without executing
    --aggressive           More thorough cleanup (higher risk)
    --type=TYPE            Project type (python|julia|mixed|auto)
    --interactive          Interactive mode with confirmation (default)
    --no-interactive       Run without interactive prompts
    --backup               Create backups before cleanup (default)
    --no-backup            Skip backup creation
    --git-integration      Analyze git repository status (default)
    --no-git-integration   Skip git analysis
    --report               Generate detailed report (default)
    --no-report            Skip report generation

EXAMPLES:
    # Basic interactive cleanup
    clean-project

    # Analyze Python project with dry-run
    clean-project --type=python --dry-run --report

    # Julia project cleanup with git integration
    clean-project --type=julia --git-integration --backup

    # Mixed project automated cleanup
    clean-project --type=mixed --no-interactive --aggressive

    # Full analysis with comprehensive reporting
    clean-project . --aggressive --git-integration --report

PROJECT TYPES:
    python    - Python scientific computing projects
    julia     - Julia scientific computing projects
    mixed     - Mixed Python/Julia projects
    auto      - Automatic detection (default)

CLEANUP CATEGORIES:
    • File Duplicates (exact content matches)
    • Unused Files (unreferenced data, modules, configs)
    • Obsolete Artifacts (cache, build files, IDE temp files)
    • Empty Directories (truly empty or only system junk)
    • Git Issues (untracked files, LFS candidates, old branches)
    • System Junk (.DS_Store, Thumbs.db, editor temp files)

SAFETY FEATURES:
    ✅ Automatic backup creation
    ✅ Git integration with stash capability
    ✅ Dry-run mode for safe preview
    ✅ Incremental execution with validation
    ✅ Automated scripts for safe operations

For more information: https://docs.claude.com/en/docs/claude-code/clean-project
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            --aggressive)
                aggressive="true"
                shift
                ;;
            --type=*)
                project_type="${1#*=}"
                shift
                ;;
            --interactive)
                interactive="true"
                shift
                ;;
            --no-interactive)
                interactive="false"
                shift
                ;;
            --backup)
                backup="true"
                shift
                ;;
            --no-backup)
                backup="false"
                shift
                ;;
            --git-integration)
                git_integration="true"
                shift
                ;;
            --no-git-integration)
                git_integration="false"
                shift
                ;;
            --report)
                report="true"
                shift
                ;;
            --no-report)
                report="false"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            -*)
                echo "❌ Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                target_path="$1"
                shift
                ;;
        esac
    done
}

# Main entry point
main() {
    # Parse arguments
    parse_arguments "$@"

    # Execute project cleanup
    clean_project "$target_path"
}

# Execute main function when script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi