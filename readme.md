# Installation

```
pip install -e 'git+https://gwangjinkim@bitbucket.org/gwangjinkim/paths.git#egg=paths'
```

# Usage

```
from paths import p_chipseqproject

# choose pathset for your project
p = p_chipseqproject

p['genome']
p['results']
p['galaxy']
p['model']
```

```
from paths import computer
computer
```
