workflow "python syntax checker" {
  resolves = ["Python Syntax Checker"]
  on = "push"
}

action "Python Syntax Checker" {
  uses = "cclauss/Find-Python-syntax-errors-action@master"
}
