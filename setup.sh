#!/bin/bash

# Function to detect the operating system and set the appropriate shell configuration file
setup_pythonpath() {
    PYTHONPATH_EXPORT="export PYTHONPATH=\"$(pwd)\""
    
    case "$(uname -s)" in
        Linux*)
            # Linux systems (write to .bashrc)
            SHELL_CONFIG="$HOME/.bashrc"
            echo "$PYTHONPATH_EXPORT" >> "$SHELL_CONFIG"
            echo "PYTHONPATH added to .bashrc"
            source "$SHELL_CONFIG"
            ;;
        Darwin*)
            # macOS systems (write to .zshrc)
            SHELL_CONFIG="$HOME/.zshrc"
            echo "$PYTHONPATH_EXPORT" >> "$SHELL_CONFIG"
            echo "PYTHONPATH added to .zshrc"
            source $SHELL_CONFIG
            ;;
        MINGW*|MSYS*|CYGWIN*)
            # Windows systems (via Git Bash or WSL, write to .bashrc)
            SHELL_CONFIG="$HOME/.bashrc"
            echo "$PYTHONPATH_EXPORT" >> "$SHELL_CONFIG"
            echo "PYTHONPATH added to .bashrc for Windows"
            source "$SHELL_CONFIG"
            ;;
        *)
            echo "Unsupported OS, please set PYTHONPATH manually."
            exit 1
            ;;
    esac
}

# Step 1: Set the PYTHONPATH to the working directory and run the appropriate shell configuration
setup_pythonpath

# Install dependencies
# pip install -r requirements.txt

# Step 2: Set up ZenML stack
bash scripts/create_zenml_stack.sh

# Step 3: Run ZenML server
bash scripts/run_zenml_server.sh

# Confirmation message
echo "Setup complete!"
echo "You can now run 'bash run.sh' to run the web app."
echo "To stop the ZenML server, run 'bash shutdown.sh'."
