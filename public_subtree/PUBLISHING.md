# Publishing the public subtree

This document shows how to publish the `public_subtree` folder to a public GitHub repository.

Default remote URL baked into the script:

https://github.com/AadarshCanCode/CombinatorialReasoning.git

From the repository root (ensure your working tree is clean):

```powershell
# Run with defaults (uses the repo URL above)
.\public_subtree\publish_subtree.ps1

# Or provide explicitly
.\public_subtree\publish_subtree.ps1 -RemoteUrl "https://github.com/your-org/your-repo.git" -Branch "main"
```

Notes:
- The script uses `git subtree split` to extract the `public_subtree` folder and pushes it to the target remote.
- Make sure you have push access to the target repo.
- The script will create and then delete a temporary local branch called `publish-subtree-branch`.
