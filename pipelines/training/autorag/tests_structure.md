

The components and pipelines are tested on the following planes: 

- unit tests
  Run locally, small-scoped, test the logic and output of the functions under different input conditions

- integration tests
  Run mostly locally, test proper behaviour of the different components used and their inter-operability. Mostly:
    - ai4rag API,
    - S3 storage upload / download, 
    - llama-stack server vector store spin-up, etc.
    - Openshift cluster deployed models' APIs,
    - Kubeflow Pipelines APIs, 
    - 

- end-to-end tests
  Run on the deployed Openshift cluster resembling the production deployment. Test whole non-mocked pipelines execution and the produced artifacts

- regression tests
  These exist in the form of additional asserts and checks performed after end-to-end tests suite. They include:
    - overal execution time validation with a small allowed margin, 
    - produced patterns' quality checks (each run should be reproducible to a certain extent) and quality should not decrease significantly with newer product releases,
    - system resources' usage comparison against the previous suite's run, 
    - 

- negative test cases
  Mainly incorporated in all of the above defined suites. They focus on users' experience in cases things go wrong (clear and intuitive errors) and error handling in general,

  