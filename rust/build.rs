// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//! Embed the git version, like hatch-vcs does for the Python package.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    for path in ["HEAD", "index"] {
        if let Some(path) = git(&["rev-parse", "--git-path", path]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rerun-if-changed=src");
    let version = git(&["describe", "--tags", "--always", "--dirty"])
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SVSBENCH_GIT_VERSION={version}");
}
