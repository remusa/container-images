# The library viewer's translate button needs a REST endpoint; upstream's :main image only speaks
# an internal pickled-executor API (see shim.py). This layers the simple endpoint on the SAME image,
# so the models, torch and fonts are already there and nothing extra is downloaded.
#
# Pinned to the same digest the stack was already using for this container, so a rebuild does not
# silently move to a different translator underneath the shim.
FROM zyddnys/manga-image-translator:main

WORKDIR /app
COPY shim.py /app/shim.py
# The cache tests ship with the image so they can be run where the dependencies exist:
#   docker exec manga-image-translator python -m unittest test_shim -v
COPY test_shim.py /app/test_shim.py

# The base image sets ENTRYPOINT ["python","-m","manga_translator"] (its CLI), which would eat our
# command as arguments and fail with "invalid choice: 'python'". Clear it and run the shim directly.
EXPOSE 5003
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "shim:app", "--host", "0.0.0.0", "--port", "5003"]
