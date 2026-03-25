## research

```
      scope
      |
  plan-experiment
      |
      +--- simple:
      |
      |  implement-experiment
      |
      +--- setup phases:
      |
      |  make-groups
      |  |
      |  +----+ FOR EACH PHASE:
      |  |    |
      |  |    make-plan --- implement
      |  |
      |  +----+
      |
  run-experiment <-> [x fail -> adjust]
      |
  write-report
      |
    test <-> [x fail -> fix]
```
