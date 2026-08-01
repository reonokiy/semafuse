//! Cross-platform discovery and invocation of agent executables.
//!
//! Windows package managers commonly expose JavaScript CLIs as `.cmd` or
//! `.bat` shims. Those files are not native executables, so callers must run
//! them through `cmd.exe`; native executables and POSIX commands are launched
//! directly.

use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
};

/// A fully resolved process program and argument vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    /// Native executable to pass to `std::process::Command` or
    /// `tokio::process::Command`.
    pub program: PathBuf,
    /// Arguments to pass to `program`.
    pub args: Vec<OsString>,
    /// Windows command text appended without C-runtime argv escaping.
    ///
    /// This is used only for the command following `cmd.exe /d /s /c`.
    pub raw_arg: Option<OsString>,
}

impl CommandSpec {
    /// Builds a standard-library process command from this specification.
    pub fn std_command(self) -> std::process::Command {
        let mut command = std::process::Command::new(self.program);
        command.args(self.args);
        if let Some(raw_arg) = self.raw_arg {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                command.raw_arg(raw_arg);
            }
            #[cfg(not(windows))]
            command.arg(raw_arg);
        }
        command
    }

    /// Builds an asynchronous Tokio process command from this specification.
    pub fn tokio_command(self) -> tokio::process::Command {
        self.std_command().into()
    }
}

/// Returns whether an executable can be resolved from the current `PATH`.
pub fn command_available(command: &str) -> bool {
    resolve_executable_in(
        command,
        std::env::var_os("PATH").as_deref(),
        std::env::var_os("PATHEXT").as_deref(),
        cfg!(windows),
    )
    .is_some()
}

/// Resolves an agent command and prepares any platform shell wrapper it needs.
pub fn command_spec<I, S>(command: &str, args: I) -> CommandSpec
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    command_spec_in(
        command,
        args,
        std::env::var_os("PATH").as_deref(),
        std::env::var_os("PATHEXT").as_deref(),
        std::env::var_os("COMSPEC").as_deref(),
        cfg!(windows),
    )
}

/// Resolves a command using explicit platform inputs.
///
/// This is public to support platform-behavior tests on non-Windows hosts.
#[doc(hidden)]
pub fn resolve_executable_in(
    command: &str,
    paths: Option<&OsStr>,
    path_ext: Option<&OsStr>,
    windows: bool,
) -> Option<PathBuf> {
    let command_path = Path::new(command);
    let has_directory = command_path.is_absolute()
        || command_path
            .parent()
            .is_some_and(|parent| !parent.as_os_str().is_empty());
    let extensions = executable_extensions(command_path, path_ext, windows);

    if has_directory {
        return extensions
            .iter()
            .find_map(|extension| existing_candidate(command_path, extension, windows));
    }

    let paths = paths?;
    std::env::split_paths(paths).find_map(|directory| {
        extensions
            .iter()
            .find_map(|extension| existing_candidate(&directory.join(command), extension, windows))
    })
}

/// Builds a command specification using explicit platform inputs.
///
/// This is public to support platform-behavior tests on non-Windows hosts.
#[doc(hidden)]
pub fn command_spec_in<I, S>(
    command: &str,
    args: I,
    paths: Option<&OsStr>,
    path_ext: Option<&OsStr>,
    command_processor: Option<&OsStr>,
    windows: bool,
) -> CommandSpec
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let executable = resolve_executable_in(command, paths, path_ext, windows)
        .unwrap_or_else(|| PathBuf::from(command));
    let args = args
        .into_iter()
        .map(|value| value.as_ref().to_os_string())
        .collect::<Vec<_>>();

    if windows && is_command_script(&executable) {
        let shell_args = vec![
            OsString::from("/d"),
            OsString::from("/s"),
            OsString::from("/c"),
        ];
        CommandSpec {
            program: command_processor
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("cmd.exe")),
            args: shell_args,
            raw_arg: Some(OsString::from(cmd_invocation(&executable, &args))),
        }
    } else {
        CommandSpec {
            program: executable,
            args,
            raw_arg: None,
        }
    }
}

fn executable_extensions(command: &Path, path_ext: Option<&OsStr>, windows: bool) -> Vec<String> {
    if !windows || command.extension().is_some() {
        return vec![String::new()];
    }
    let configured = path_ext
        .and_then(OsStr::to_str)
        .into_iter()
        .flat_map(|value| value.split(';'))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            if value.starts_with('.') {
                value.to_owned()
            } else {
                format!(".{value}")
            }
        })
        .collect::<Vec<_>>();
    if configured.is_empty() {
        vec![".COM".into(), ".EXE".into(), ".BAT".into(), ".CMD".into()]
    } else {
        configured
    }
}

fn existing_candidate(base: &Path, extension: &str, windows: bool) -> Option<PathBuf> {
    let candidate = if extension.is_empty() {
        base.to_path_buf()
    } else {
        let mut candidate = base.as_os_str().to_os_string();
        candidate.push(extension);
        PathBuf::from(candidate)
    };
    if candidate.is_file() {
        return Some(candidate);
    }
    if !windows {
        return None;
    }

    let parent = candidate
        .parent()
        .filter(|path| !path.as_os_str().is_empty())?;
    let expected = candidate.file_name()?.to_string_lossy();
    fs::read_dir(parent).ok()?.flatten().find_map(|entry| {
        let actual = entry.file_name();
        (actual.to_string_lossy().eq_ignore_ascii_case(&expected)
            && entry.file_type().is_ok_and(|kind| kind.is_file()))
        .then(|| entry.path())
    })
}

fn is_command_script(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
        })
}

fn cmd_invocation(executable: &Path, args: &[OsString]) -> String {
    let mut tokens = Vec::with_capacity(args.len() + 1);
    tokens.push(cmd_quote(executable.as_os_str()));
    tokens.extend(args.iter().map(|argument| cmd_quote(argument)));
    // With `/s /c`, cmd.exe removes exactly this outer quote pair and retains
    // the quotes around every individual token, including a shim path with
    // spaces or shell metacharacters.
    format!("\"{}\"", tokens.join(" "))
}

fn cmd_quote(value: &OsStr) -> String {
    format!("\"{}\"", value.to_string_lossy().replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn windows_command_shims_use_the_command_processor() {
        let temp = TempDir::new().unwrap();
        let shim = temp.path().join("codex.CMD");
        fs::write(&shim, b"").unwrap();
        let spec = command_spec_in(
            "codex",
            ["app-server", "--listen", "stdio://"],
            Some(temp.path().as_os_str()),
            Some(OsStr::new(".EXE;.CMD")),
            Some(OsStr::new(r"C:\Windows\System32\cmd.exe")),
            true,
        );

        assert_eq!(spec.program, Path::new(r"C:\Windows\System32\cmd.exe"));
        assert_eq!(
            &spec.args[..3],
            [
                OsString::from("/d"),
                OsString::from("/s"),
                OsString::from("/c")
            ]
        );
        let invocation = spec.raw_arg.as_ref().unwrap().to_string_lossy();
        assert!(invocation.starts_with("\"\""));
        assert!(invocation.contains("codex.CMD\" \"app-server\""));
        assert!(invocation.ends_with("\"\""));
    }

    #[test]
    fn native_executables_are_started_directly() {
        let temp = TempDir::new().unwrap();
        let executable = temp.path().join("codex.EXE");
        fs::write(&executable, b"").unwrap();
        let spec = command_spec_in(
            "codex",
            ["app-server"],
            Some(temp.path().as_os_str()),
            Some(OsStr::new(".EXE;.CMD")),
            None,
            true,
        );

        assert_eq!(spec.program, executable);
        assert_eq!(spec.args, [OsString::from("app-server")]);
        assert!(spec.raw_arg.is_none());
    }

    #[cfg(windows)]
    #[test]
    fn command_script_spec_executes_through_cmd() {
        let temp = TempDir::new().unwrap();
        let shim = temp.path().join("opencode.cmd");
        let capture = temp.path().join("capture.txt");
        fs::write(
            &shim,
            "@echo off\r\n>\"%~dp0capture.txt\" echo %*\r\nexit /b 0\r\n",
        )
        .unwrap();
        let spec = command_spec_in(
            "opencode",
            ["serve", "--port", "4321"],
            Some(temp.path().as_os_str()),
            Some(OsStr::new(".CMD")),
            std::env::var_os("COMSPEC").as_deref(),
            true,
        );

        let status = spec.std_command().status().unwrap();
        assert!(status.success());
        assert_eq!(
            fs::read_to_string(capture).unwrap().trim(),
            "serve --port 4321"
        );
    }
}
