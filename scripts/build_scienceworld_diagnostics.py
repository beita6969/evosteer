"""Build a separate read-only diagnostic overlay; never modify the deployed jar.

The missing first-taskFailure branch was observed in A17. Add logging fields
only: do not replace predicates, scoring, parser, physical state or actions.
Compile just the two changed Scala files against the existing released jar.
Compiler artifacts come from the official Scala distribution on Maven Central.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

GOAL = "simulator/src/main/scala/scienceworld/tasks/goals/Goal.scala"
INTERFACE = "simulator/src/main/scala/scienceworld/runtime/AgentInterface.scala"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("diagnostic insertion point differs from supported upstream source")
    return source.replace(old, new, 1)


def instrument_goal(source: str) -> str:
    source = replace_once(
        source,
        "  var failed:Boolean = false",
        """  var failed:Boolean = false
  var diagnosticFailureBranch:String = ""
  var diagnosticFailureGoalType:String = ""
  var diagnosticFailureGoalIndex:Int = -1
  var diagnosticFailureTickPhase:String = ""
  var diagnosticFailureMove:Int = -1
  var diagnosticTickPhase:String = ""
  var diagnosticMove:Int = -1""",
    )
    source = replace_once(
        source,
        "    this.failed = false",
        """    this.failed = false
    this.diagnosticFailureBranch = ""
    this.diagnosticFailureGoalType = ""
    this.diagnosticFailureGoalIndex = -1
    this.diagnosticFailureTickPhase = ""
    this.diagnosticFailureMove = -1""",
    )
    source = replace_once(
        source,
        '        //println ("Task failure.")\n        this.setFailed()',
        """        if (!this.failed) {
          this.diagnosticFailureBranch = goalReturn.diagnosticFailureBranch
          this.diagnosticFailureGoalType = curSubgoal.get.getClass.getName
          this.diagnosticFailureGoalIndex = subgoalIdx
          this.diagnosticFailureTickPhase = this.diagnosticTickPhase
          this.diagnosticFailureMove = this.diagnosticMove
        }
        this.setFailed()""",
    )
    return replace_once(
        source,
        "class GoalReturn(val subgoalSuccess:Boolean, val taskFailure:Boolean) {",
        '''class GoalReturn(val subgoalSuccess:Boolean, val taskFailure:Boolean) {
  val diagnosticFailureBranch:String = if (taskFailure) {
    Thread.currentThread().getStackTrace.filter(_.getClassName.startsWith("scienceworld."))
      .map(_.toString).mkString("\\n")
  } else ""''',
    )


def instrument_interface(source: str) -> str:
    source = replace_once(
        source,
        "  private var curIter:Int = 0",
        '  private var curIter:Int = 0\n  var diagnosticResolvedAction:String = ""',
    )
    source = replace_once(
        source,
        "    // Parse user input\n",
        '    diagnosticResolvedAction = ""\n    // Parse user input\n',
    )
    source = replace_once(
        source,
        "    val (success, statusStr) = this.processUserInput(userInputStr, universe)",
        """    val (success, statusStr) = this.processUserInput(userInputStr, universe)
    diagnosticResolvedAction = actionHandler.queuedActions.map(a =>
      a.getClass.getName + ": " + a.assignments.toSeq.sortBy(_._1).map {
        case (key, obj) => key + "=" + obj.name + "#" + obj.uuid
      }.mkString(", ")).mkString("; ")""",
    )
    old = "          task.goalSequence.tick(objMonitor, agent)"
    if source.count(old) != 2:
        raise ValueError("native pre/post physics goal evaluation sites changed")
    end = source.index("def step(")
    for phase in ("before-physics", "after-physics"):
        # Replace a different uninstrumented occurrence each time.
        index = source.index(old, end)
        inserted = (
            f'          task.goalSequence.diagnosticTickPhase = "{phase}"\n'
            "          task.goalSequence.diagnosticMove = this.curIter\n" + old
        )
        source = source[:index] + inserted + source[index + len(old) :]
        end = index + len(inserted)
    return source


def build(source_root: Path, original_jar: Path, destination: Path, java: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    compiler_dir = destination / "compiler"
    compiler_dir.mkdir()
    build_text = (source_root / "simulator/build.sbt").read_text()
    match = re.search(r'scalaVersion\s*:=\s*"([\d.]+)"', build_text)
    if match is None:
        raise ValueError("upstream does not declare its Scala compiler version")
    version = match[1]
    jars = []
    for name in ("scala-compiler", "scala-library", "scala-reflect"):
        target = compiler_dir / f"{name}-{version}.jar"
        url = f"https://repo.maven.apache.org/maven2/org/scala-lang/{name}/{version}/{target.name}"
        with (
            urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310 - fixed HTTPS origin
            target.open("xb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        jars.append(str(target))
    sources = []
    for path, transform in ((GOAL, instrument_goal), (INTERFACE, instrument_interface)):
        target = destination / Path(path).name
        target.write_text(transform((source_root / path).read_text()))
        sources.append(str(target))
    classes = destination / "classes"
    classes.mkdir()
    subprocess.run(  # noqa: S603 - explicit operator-selected Java; no shell
        [
            str(java),
            "-Dscala.usejavacp=true",
            "-cp",
            ":".join(jars),
            "scala.tools.nsc.Main",
            "-target:jvm-1.8",
            "-classpath",
            str(original_jar),
            "-d",
            str(classes),
            *sources,
        ],
        check=True,
        timeout=180,
    )
    replacements = {str(path.relative_to(classes)): path for path in classes.rglob("*.class")}
    result = destination / "scienceworld-diagnostics.jar"
    with (
        zipfile.ZipFile(original_jar) as src,
        zipfile.ZipFile(result, "x", zipfile.ZIP_DEFLATED) as out,
    ):
        for info in src.infolist():
            if info.filename not in replacements:
                out.writestr(info, src.read(info.filename))
        for name, class_path in replacements.items():
            out.write(class_path, name)
    (destination / "build-private.json").write_text(
        json.dumps(
            {
                "profile": "scienceworld-diagnostic-overlay@1",
                "source_root": str(source_root),
                "original_jar": str(original_jar),
                "scala_version": version,
                "modified_sources": [GOAL, INTERFACE],
                "replaced_classes": sorted(replacements),
                "output_jar": str(result),
                "original_files_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--original-jar", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--java", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source_root, args.original_jar, args.destination, args.java))
