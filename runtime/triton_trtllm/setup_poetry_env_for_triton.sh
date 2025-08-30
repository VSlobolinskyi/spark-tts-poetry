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
    
    # Handle TensorRT-LLM Python module and extract additional symbols
    if [ -d "$POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_llm" ]; then
        echo "Finding all TensorRT-LLM shared objects for symbol resolution..."
        TRTLLM_SO_FILES=$(find "$POETRY_ENV/lib/python$PYTHON_VERSION/site-packages/tensorrt_llm" -name "*.so" -type f 2>/dev/null)
        
        # Create a directory for additional TensorRT-LLM libraries
        mkdir -p /content/tritonserver/lib/tensorrt_llm_extra
        
        # Copy all TensorRT-LLM .so files to ensure all symbols are available
        for so_file in $TRTLLM_SO_FILES; do
            base_name=$(basename "$so_file")
            if [ ! -f "/content/tritonserver/lib/tensorrt_llm/$base_name" ]; then
                echo "Copying additional TensorRT-LLM library: $so_file"
                cp -v "$so_file" /content/tritonserver/lib/tensorrt_llm_extra/
            fi
        done
        
        # Add this directory to LD_LIBRARY_PATH
        export LD_LIBRARY_PATH="/content/tritonserver/lib/tensorrt_llm_extra:$LD_LIBRARY_PATH"
        
        # Create links in the Triton backends directory for any missing libraries
        if [ -d "/content/tritonserver/backends/tensorrtllm" ]; then
            echo "Creating links in TensorRT-LLM backend directory..."
            for lib in /content/tritonserver/lib/tensorrt_llm/*.so /content/tritonserver/lib/tensorrt_llm_extra/*.so; do
                if [ -f "$lib" ]; then
                    base_name=$(basename "$lib")
                    if [ ! -f "/content/tritonserver/backends/tensorrtllm/$base_name" ]; then
                        ln -sf "$lib" "/content/tritonserver/backends/tensorrtllm/$base_name"
                        echo "Created backend link: /content/tritonserver/backends/tensorrtllm/$base_name -> $lib"
                    fi
                fi
            done
        fi
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
    
    # Add special handling for the tensorrtllm backend
    echo "Setting up TensorRT-LLM backend compatibility..."
    
    # If the backend directory exists, add special handling
    if [ -d "/content/tritonserver/backends/tensorrtllm" ]; then
        echo "Found TensorRT-LLM backend, setting up additional compatibility..."
        
        # First, check if the library exists
        if [ -f "/content/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so" ]; then
            # Get dependencies of the backend library
            echo "Checking TensorRT-LLM backend dependencies..."
            ldd /content/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so 2>&1 | grep "not found" | awk '{print $1}' > /tmp/missing_libs.txt
            
            if [ -s /tmp/missing_libs.txt ]; then
                echo "Found missing dependencies:"
                cat /tmp/missing_libs.txt
                echo "Attempting to resolve missing dependencies..."
                
                # Find all libraries with similar names in the poetry environment
                while IFS= read -r lib; do
                    # Strip leading 'lib' and trailing '.so'
                    base_lib=$(echo "$lib" | sed 's/^lib//' | sed 's/\.so.*//')
                    echo "Searching for library containing $base_lib..."
                    
                    # Find potential libraries and create symlinks
                    find "$POETRY_ENV" -name "*$base_lib*.so*" -type f 2>/dev/null | while read -r found_lib; do
                        found_base=$(basename "$found_lib")
                        echo "Found potential match: $found_lib"
                        cp -v "$found_lib" /content/tritonserver/lib/
                        
                        # Create symlink with exact name being looked for
                        ln -sf "/content/tritonserver/lib/$found_base" "/content/tritonserver/lib/$lib"
                        echo "Created symlink: /content/tritonserver/lib/$lib -> /content/tritonserver/lib/$found_base"
                    done
                done < /tmp/missing_libs.txt
            else
                echo "No missing dependencies found in ldd output"
            fi
        else
            echo "Warning: TensorRT-LLM backend library not found at /content/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so"
        fi
    fi
    
    # Special handling for the specific symbol from the error
    echo "Adding special handling for TensorRT-LLM scheduler symbols..."
    find "$POETRY_ENV" -name "*tensorrt_llm*executor*.so*" -type f 2>/dev/null | while read -r found_lib; do
        echo "Found TensorRT-LLM executor library: $found_lib"
        cp -v "$found_lib" /content/tritonserver/lib/
        
        # Create symlinks to ensure this library is found first
        ln -sf "$found_lib" "/content/tritonserver/backends/tensorrtllm/$(basename "$found_lib")"
        echo "Created symlink in backend directory"
    done
    
    # Check for specific Python modules with TensorRT-LLM dependencies
    if python -c "import tensorrt_llm.executor" 2>/dev/null; then
        echo "Found tensorrt_llm.executor module, exporting path for symbols..."
        EXECUTOR_PATH=$(python -c "import tensorrt_llm.executor, os; print(os.path.dirname(tensorrt_llm.executor.__file__))" 2>/dev/null)
        
        if [ -n "$EXECUTOR_PATH" ]; then
            echo "TensorRT-LLM executor module path: $EXECUTOR_PATH"
            export LD_LIBRARY_PATH="$EXECUTOR_PATH:$LD_LIBRARY_PATH"
            
            # Copy any .so files from this path
            find "$EXECUTOR_PATH" -name "*.so" -type f 2>/dev/null | while read -r lib; do
                echo "Copying executor library: $lib"
                cp -v "$lib" /content/tritonserver/lib/
                ln -sf "/content/tritonserver/lib/$(basename "$lib")" "/content/tritonserver/backends/tensorrtllm/$(basename "$lib")"
            done
        fi
    fi
    
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