#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    const char *home = getenv("HOME");
    if (!home) { fprintf(stderr, "HOME not set\n"); return 1; }

    char appdata[1024], python[1024], script[1024], logpath[1024];
    snprintf(appdata,  sizeof(appdata),  "%s/Library/Application Support/WhisperFlow", home);
    snprintf(python,   sizeof(python),   "%s/venv/bin/python",  appdata);
    snprintf(script,   sizeof(script),   "%s/src/app.py",       appdata);
    snprintf(logpath,  sizeof(logpath),  "%s/whisperflow.log",  appdata);

    if (chdir(appdata) != 0) {
        fprintf(stderr, "WhisperFlow not installed — run install.sh first\n");
        return 1;
    }

    /* Redirect stdout+stderr to log file */
    freopen(logpath, "w", stdout);
    freopen(logpath, "a", stderr);

    execl(python, python, script, (char *)NULL);
    fprintf(stderr, "Failed to exec %s\n", python);
    return 1;
}
