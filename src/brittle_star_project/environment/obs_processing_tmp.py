obs = {}
segs = set()
joints = set()

for key, arr in obs.items():
    if key in segs:
        # padding using mask 1x
        pass
    elif key in joints:
        # padding using mask 2x
        pass
    else:
        # no padding
        pass

    # reshape according to segs, joints, or global
    pass
