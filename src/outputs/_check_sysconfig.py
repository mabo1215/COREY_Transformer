import sysconfig, sys
print("Python:", sys.executable)
print("include:", sysconfig.get_path("include"))
print("platinclude:", sysconfig.get_path("platinclude"))
