# Contributing

1. Pick or open an issue and assign it to yourself.
2. Branch from `main`. For example, use `feature/breathing-filter` for a change to the breathing filter.
3. Use the [README setup](README.md#setup), then run `python -m ruff check .` and `python -m pytest`. Update tests when you change behavior.
4. Open a pull request, describe how you tested the change, and have a teammate review it.

Keep secrets, recordings, trial files, and identifiable health data out of commits. Label synthetic data and results wherever you use them.

## Browser changes

Run `python docs/build_web_core.py` after changing the shared Python chest tracker, breathing estimator, assessment logic, hand-response logic, or monitor in `src/ostinato`. Include the regenerated `docs/live-core.json` in your change. The browser engine and worker live in `docs/`.

Serve `docs/` using the [local preview command](README.md#try-it-in-the-browser). Exercise **Try sample**, check a hand response, and verify **Stop** returns the app to its inactive state. If you have authorized webcam access, also exercise camera permission, framing guidance, and camera shutdown on **Stop**. See the [demo guide](docs/DEMO.md) for a hands-on walkthrough and the [paired trial guide](docs/VIEWPOINT_TRIALS.md) for measuring real-camera accuracy. The sample and automated tests cannot establish accuracy on people.

## Publish the public site

The source repository is [lay-shah/ostinato](https://github.com/lay-shah/ostinato). Its CI runs lint and tests. The [sahilnpatel.github.io repository](https://github.com/sahilnpatel/sahilnpatel.github.io) publishes the public site from its `main` branch. Merging the source PR alone does not deploy the site.

1. Check out the approved source revision. The staging command uses the committed files at `HEAD`. Stage the public site into a directory outside the source checkout, for example with `python tools/build_site.py /tmp/ostinato-site`. You can also use the `ostinato-site` artifact from CI for that revision.
2. Update the `ostinato/` directory in the Pages repository with the staged files. These include the landing page, project report, live page, browser assets, and `release.json` with the source revision and asset hashes. Review the diff and merge the Pages change to `main`.
3. In the Pages repository’s Actions tab, verify that **Deploy GitHub Pages** succeeds after the push to `main`. You can also run it manually. Publishing requires GitHub access to the publishing repository.
4. Open the [published live page](https://sahilnpatel.github.io/ostinato/live.html) and run the sample. Check the [release manifest](https://sahilnpatel.github.io/ostinato/release.json) for the intended source revision.
