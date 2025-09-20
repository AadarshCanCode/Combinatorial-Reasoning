# PowerShell script to publish the public_subtree to a public repository using git subtree
# Usage: .\publish_subtree.ps1 -RemoteUrl "https://github.com/your-org/public-crqubo.git" -Branch "main"
param(
    [string]$RemoteUrl = 'https://github.com/AadarshCanCode/CombinatorialReasoning.git',

    [string]$Branch = 'main',

    [string]$SubtreeDir = 'public_subtree'
)

Write-Host "Preparing to publish subtree '$SubtreeDir' to remote $RemoteUrl (branch: $Branch)"

# Ensure working tree is clean
git update-index -q --refresh
$changes = git status --porcelain
if ($changes) {
    Write-Error "Working tree not clean. Commit or stash changes before publishing."
    exit 1
}

# Add remote if missing
$existing = git remote
if (-not ($existing -contains 'public')) {
    git remote add public $RemoteUrl
} else {
    git remote set-url public $RemoteUrl
}

# Create subtree split
$split = git subtree split --prefix=$SubtreeDir -b publish-subtree-branch
if ($LASTEXITCODE -ne 0) {
    Write-Error "git subtree split failed"
    exit 1
}

# Push the split branch to the remote
git push public publish-subtree-branch:$Branch --force
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed"
    exit 1
}

# Cleanup
git branch -D publish-subtree-branch
Write-Host "Subtree published to $RemoteUrl on branch $Branch"
