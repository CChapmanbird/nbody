class axis_handle:
    bodies=None
    lines=None
    com=None

    def __init__(self, view='position', zoom=0, zoomfocus=0, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title='', grid='on'):
        self.view = view
        self.zoom = zoom
        self.zoomfocus = zoomfocus
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.grid = grid
