"""Functions related to jupyter kernels/notebooks.

Code based on
https://stackoverflow.com/questions/12544056/how-do-i-get-the-current-ipython-jupyter-notebook-name
"""
import os


def get_kernel_id():
    """Return the kernel id."""
    import ipykernel

    connection_file = os.path.basename(ipykernel.get_connection_file())
    return connection_file.split('-', 1)[1].split('.')[0]


def notebook_path():
    """
    Returns the absolute path of the Notebook or None if it cannot be
    determined.

    NOTE:
        works only when the security is token-based or if there is also
        no password set.
    """
    from notebook import notebookapp
    import urllib
    import json

    kernel_id = get_kernel_id()

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token'] == '' and not srv['password']:
                # No token and no password, ahem...
                url = srv['url'] + 'api/sessions'
            else:
                url = srv['url'] + 'api/sessions?token=' + srv['token']

            req = urllib.request.urlopen(url)
            sessions = json.load(req)

            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(
                        srv['notebook_dir'],
                        sess['notebook']['path']
                    )

        except Exception:
            pass  # There may be stale entries in the runtime directory

    return None
