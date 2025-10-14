# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version being used, the version of autogluon, scikitlearn, pytorch as is relevant
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

Ideally, you can install OxonFair and its dependencies in a fresh virtualenv to reproduce the bug.

## Contributing via Pull Requests
Code contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *master* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source (see details below); please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Tips for Modifying the Source Code

- Use a fresh virtualenv.


- Please try to avoid introducing additional dependencies on 3rd party packages. We are currently working to reduce the number of external dependencies of our package.

- All code should adhere to the [PEP8 style](https://www.python.org/dev/peps/pep-0008/).

- After you have edited the code, ensure your changes pass the unit tests via:
...

- We encourage you to add your own unit tests, but please ensure they run quickly (unit tests should train models on small data-subsample with the lowest values of training iterations and time-limits that suffice to evaluate the intended functionality). You can run a specific unit test within a specific file like this:
```
python3 -m pytest path_to_file::test_mytest
```
Or remove the ::test_mytest suffix to run all tests in the file:
```
python3 -m pytest path_to_file
```

- To otherwise test your code changes, we recommend running OxonFair on multiple datasets and verifying the code runs correctly and the resulting accuracy of the trained models is not harmed by your change.  One easy way to test is to simply modify the scripts in `examples/`.

- Remember to update all existing examples/tutorials/documentation affected by your code changes.

- After you open your pull request, our CI system will run for little while to check your code and report found errors. Please check back and fix any errors encountered at this stage (you can retrigger a new CI check by pushing updated code to the same PR in a new commit).



## Finding Contributions to Work On
Looking at the existing issues is a great way to find something to contribute on. As our project uses the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted') issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.



## Licensing

See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
