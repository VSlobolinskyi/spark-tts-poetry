#!/bin/bash
# Helper script to set up Poetry environment for Triton
# This script comprehensively maps the Poetry environment to make it available to Triton

setup_poetry_env_for_triton() {
    echo "Setting up Poetry environment for Triton..."
    
    # Check if Poetry is available
    if ! command -v poetry &> /dev/null; then
        echo "Poetry not found, using default environment"
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        export LD_LIBRARY_PATH="/content/tritonserver/lib:/content/tritonserver/lib/stubs:$LD_LIBRARY_PATH"
        return
    fi
    
    # Get Poetry environment path
    POETRY_ENV=$(poetry env info -p)
    if [ -z "$POETRY_ENV" ]; then
        echo "Error: Failed to get Poetry environment path"
        return
    fi
    echo "Poetry environment: $POETRY_ENV"
    
    # Get Python version used by Poetry
    PYTHON_VERSION=$(poetry run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ -z "$PYTHON_VERSION" ]; then
        echo "Warning: Could not determine Python version, defaulting to 3.10"
        PYTHON_VERSION="3.10"
    fi
    echo "Poetry uses Python $PYTHON_VERSION"
    
    # DO NOT set PYTHONHOME as it causes initialization errors with Triton
    
    # Set PATH to include Poetry's bin directory
    export PATH="$POETRY_ENV/bin:$PATH"
    echo "Added Poetry's bin to PATH"
    
    # Store the Poetry Python path for Triton config
    POETRY_PYTHON_PATH="$POETRY_ENV/bin/python"
    
    # Set PYTHONPATH to include Poetry's site-packages and Python lib directories
    export PYTHONPATH="$POETRY_ENV/lib/python$PYTHON_VERSION/site-packages:$POETRY_ENV/lib/python$PYTHON_VERSION:$PROJECT_ROOT:$PYTHONPATH"
    echo "Set PYTHONPATH to include Poetry's Python libraries"
    
    # Find all directories containing library files in the Poetry environment
    echo "Finding all library directories in Poetry environment..."
    LIB_DIRS=""
    # Look for more library file extensions
    while IFS= read -r dir; do
        if [ -n "$dir" ]; then
            LIB_DIRS="$dir:$LIB_DIRS"
            echo "Found library directory: $dir"
        fi
    done < <(find "$POETRY_ENV" \( -name "*.so" -o -name "*.so.*" -o -name "*.a" -o -name "*.la" \) -type f 2>/dev/null | xargs -r dirname 2>/dev/null | sort | uniq)
    
    # Add all lib directories to LD_LIBRARY_PATH
    if [ -n "$LIB_DIRS" ]; then
        export LD_LIBRARY_PATH="$LIB_DIRS$POETRY_ENV/lib:/content/tritonserver/lib:/content/tritonserver/lib/stubs:$LD_LIBRARY_PATH"
        echo "Added all library directories to LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$POETRY_ENV/lib:/content/tritonserver/lib:/content/tritonserver/lib/stubs:$LD_LIBRARY_PATH"
        echo "Added Poetry's lib directory to LD_LIBRARY_PATH"
    fi
    
    # Create directories for TensorRT-LLM libraries
    mkdir -p /content/tritonserver/lib/tensorrt_llm
    mkdir -p /content/tritonserver/lib/tensorrt
    
    # Handle TensorRT-LLM libraries
    if [ -d "$POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_llm/libs" ]; then
        echo "Copying TensorRT-LLM libraries to Triton lib directory..."
        cp -v $POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_llm/libs/* /content/tritonserver/lib/tensorrt_llm/ 2>/dev/null
        
        # Create symbolic links with version numbers
        echo "Creating symbolic links with version numbers for TensorRT-LLM libraries..."
        cd /content/tritonserver/lib/tensorrt_llm
        for lib in *.so; do
            if [ -f "$lib" ]; then
                # Create version-specific links for TensorRT libraries
                for version in {8..12}; do
                    if [ ! -f "${lib}.${version}" ]; then
                        ln -sf "$lib" "${lib}.${version}"
                        echo "Created symlink: ${lib}.${version} -> $lib"
                    fi
                done
            fi
        done
        cd - > /dev/null
        
        # Add this directory to LD_LIBRARY_PATH
        export LD_LIBRARY_PATH="/content/tritonserver/lib/tensorrt_llm:$LD_LIBRARY_PATH"
    else
        echo "Warning: TensorRT-LLM libs directory not found at $POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_llm/libs"
    fi
    
    # Handle TensorRT libraries (specifically libnvinfer)
    if [ -d "$POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_libs" ]; then
        echo "Found TensorRT libs directory, copying to Triton library path..."
        cp -v $POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_libs/* /content/tritonserver/lib/tensorrt/ 2>/dev/null
        
        # Create symbolic links with version numbers for these libraries too
        echo "Creating symbolic links for TensorRT libraries..."
        cd /content/tritonserver/lib/tensorrt
        for lib in *.so*; do
            if [ -f "$lib" ]; then
                # Extract base name without version
                base_name=$(echo "$lib" | sed -E 's/\.so\..*$//')
                if [ "$base_name" != "$lib" ]; then
                    # Already has version - create link without version
                    ln -sf "$lib" "${base_name}.so"
                    echo "Created symlink: ${base_name}.so -> $lib"
                else
                    # No version - create links with versions
                    for version in {7..12}; do
                        if [ ! -f "${lib}.${version}" ]; then
                            ln -sf "$lib" "${lib}.${version}"
                            echo "Created symlink: ${lib}.${version} -> $lib"
                        fi
                    done
                fi
            fi
        done
        cd - > /dev/null
        
        # Add this directory to LD_LIBRARY_PATH
        export LD_LIBRARY_PATH="/content/tritonserver/lib/tensorrt:$LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH="$POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"
    else
        echo "Warning: TensorRT libs directory not found at $POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_libs"
    fi
    
    # Look for specific TensorRT libraries throughout the environment and create links if needed
    echo "Searching for specific TensorRT libraries throughout the environment..."
    for pattern in "libnvinfer*.so*" "libnvonnxparser*.so*" "libnvparsers*.so*"; do
        while IFS= read -r lib_path; do
            if [ -f "$lib_path" ]; then
                lib_name=$(basename "$lib_path")
                target_dir="/content/tritonserver/lib"
                echo "Found $lib_name at $lib_path, linking to $target_dir"
                ln -sf "$lib_path" "$target_dir/$lib_name"
                
                # Create additional version links if needed
                base_name=$(echo "$lib_name" | sed -E 's/\.so\..*$/\.so/')
                if [ "$base_name" != "$lib_name" ] && [ ! -f "$target_dir/$base_name" ]; then
                    ln -sf "$lib_path" "$target_dir/$base_name"
                    echo "Created additional symlink: $target_dir/$base_name -> $lib_path"
                fi
            fi
        done < <(find "$POETRY_ENV" -name "$pattern" -type f 2>/dev/null)
    done
    
    # Check for CUDA libraries and add to path if found
    CUDA_DIRS=$(find "$POETRY_ENV" -path "*cuda*" -type d 2>/dev/null)
    if [ -n "$CUDA_DIRS" ]; then
        echo "Found CUDA directories, adding to library path..."
        for cuda_dir in $CUDA_DIRS; do
            export LD_LIBRARY_PATH="$cuda_dir:$LD_LIBRARY_PATH"
            echo "Added CUDA directory to path: $cuda_dir"
        done
    fi
    
    # Set Triton backend directory
    export TRITON_BACKEND_DIRECTORY=/content/tritonserver/backends
    echo "Set TRITON_BACKEND_DIRECTORY=$TRITON_BACKEND_DIRECTORY"
    
    # Update system library cache
    ldconfig 2>/dev/null || echo "ldconfig failed (might need sudo)"
    
    # Print final environment
    echo "Environment variables set for Triton:"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "TRITON_BACKEND_DIRECTORY=$TRITON_BACKEND_DIRECTORY"
    echo "Poetry Python: $POETRY_PYTHON_PATH"
    
    # Verify critical libraries are accessible
    echo "Verifying critical libraries are accessible..."
    for lib in libnvinfer.so libnvinfer_plugin.so libnvonnxparser.so; do
        if ldconfig -p 2>/dev/null | grep -q "$lib"; then
            echo "✓ $lib is accessible"
        else
            echo "! Warning: $lib may not be accessible"
        fi
    done
}

# This script can be sourced or run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly
    PROJECT_ROOT=$(pwd)
    setup_poetry_env_for_triton
fi