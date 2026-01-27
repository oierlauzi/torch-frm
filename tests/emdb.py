
import requests
import tempfile
import shutil
import os
from urllib.parse import urlparse

def _make_emdb_ftp_url(id: int) -> str:
    return f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{id:04d}/map/emd_{id:04d}.map.gz"

def fetch_emdb_map(id: int) -> str:
    url = _make_emdb_ftp_url(id)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        parsed_url = urlparse(url)
        _, ext = os.path.splitext(parsed_url.path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            return tmp_file.name
