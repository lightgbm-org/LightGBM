# Security

## Reporting Security Issues

> [!WARNING]
> Do not report security vulnerabilities through public GitHub issues!

Instead, please open a private vulnerability report in this repository, on the "Security and quality" tab.

See https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately

## Report Details

We prefer all communications to be in English.

Reports should include the following:

* reproducible example showing how the vulnerability can be exploited
* statement about the impact (including affected versions)

And we'd appreciate if they also include:

* statement about whether you are interested in implementing the fix yourself

## Disclosure Policy

This project is staffed exclusively by volunteers.
Please be patient and allow us time to respond before disclosing vulnerabilities.

We prefer to coordinate disclosure privately, and are committed to giving credit for confirmed vulnerabilities.

## Security Model

### Data Formats

Binary `Dataset` files (`.bin`) are considered trusted inputs.
I/O issues related to the handling of these files should be treated as normal bugs and reported publicly.
Do not accept untrusted LightGBM `.bin` files.

The Python package supports the `pickle` serialization format, and data in that format are considered trusted inputs.
The security risks related to `pickle` are well known and well-documented, including in the Python standard library's documentation (https://docs.python.org/3/library/pickle.html).

LightGBM-custom model text files (`.json` and `.txt`) are generally considered untrusted inputs.
Reading/writing these files and using the resulting in-memory model objects is expected to be safe.

### Network

LightGBM supports distributed training, in which multiple processes communicate over a network to coordinate model training.
All participants in such networks (indicated by parameters like `machine_list` or `machines`) are trusted.
Do not include untrusted machines in distributed training networks.
