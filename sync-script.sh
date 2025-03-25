#!/bin/bash

# Define source and destination paths
WSL_PATH="$HOME/gibbs_phenomenon_studying"
GDRIVE_PATH="/mnt/i/My Drive/programming/_00_gibbs_phenomenon_studying"

# Function to sync from WSL to Google Drive
sync_to_gdrive() {
  echo "Syncing from WSL to Google Drive..."
  rsync -avz --progress --delete --no-perms --exclude="archives" --exclude=.git --exclude="*.egg-info" "$WSL_PATH/" "$GDRIVE_PATH/"
  # if [ $? -ne 0 ]; then
  #   echo "rsync encountered errors, but continued."
  # fi
  # echo "Sync to Google Drive completed."
}

# Function to sync from Google Drive to WSL
sync_to_wsl() {
  echo "Syncing from Google Drive to WSL..."
  rsync -avz --progress --delete --no-perms --exclude="archives" --exclude=.git --exclude="*.egg-info" "$GDRIVE_PATH/" "$WSL_PATH/"
  echo "Sync to WSL completed."
}

# Function to show differences between the directories
show_diff() {
  echo "Differences between WSL and Google Drive versions:"
  diff -r --brief "$WSL_PATH" "$GDRIVE_PATH" | grep -v "Only in"
  echo "Files only in WSL:"
  diff -r --brief "$WSL_PATH" "$GDRIVE_PATH" | grep "Only in $WSL_PATH"
  echo "Files only in Google Drive:"
  diff -r --brief "$WSL_PATH" "$GDRIVE_PATH" | grep "Only in $GDRIVE_PATH"
}

# Create directories if they don't exist
mkdir -p "$WSL_PATH"
mkdir -p "$GDRIVE_PATH"

# Parse command line arguments
case "$1" in
  to-gdrive)
    sync_to_gdrive
    ;;
  to-wsl)
    sync_to_wsl
    ;;
  diff)
    show_diff
    ;;
  *)
    echo "Usage: $0 {to-gdrive|to-wsl|diff}"
    echo "  to-gdrive: Sync from WSL to Google Drive"
    echo "  to-wsl: Sync from Google Drive to WSL"
    echo "  diff: Show differences between the two directories"
    exit 1
    ;;
esac

exit 0
