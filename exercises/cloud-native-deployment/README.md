# Deployment and Rollback exercises

Run from the companion root with its existing Python environment:

```sh
python -m pytest exercises/cloud-native-deployment/test_cloud_native_deployment_acceptance.py -p no:cacheprovider
python exercises/cloud-native-deployment/cloud_native_deployment_rehearse_06.py
```

The starters for exercises 1–5 fail their designed checks; their references pass.
They supply proposed answers to correct. Exercise 5's `rehearsal-05.json` contains all the day's payment records
and the claimed rehearsal result. Diagnose the violated property and write a
reply to the release owner.

For exercise 6, keep `cloud_native_deployment_canary_06.py` unchanged. Edit the
reader in `cloud_native_deployment_target_06.py`, rehearse again and retain every
stored record. Acceptance checks the real fresh-process restart, zero requests
paid twice, and unchanged decisions on all forty public scenarios.

Exercise 6's starter fails when the closed rollback reader sees `canary_id`.
The expanded reference reads the records and preserves them, and its full acceptance passes.

`cloud_native_deployment_reference.py` is public. Acceptance imports only
the student files. To validate the reference, place its answers and reader in the
student files in an isolated exercise copy, then run the same acceptance command.

The simulated rehearsal never produces start authority. The original grant's
expiry and exhausted status survive version changes and restarts. Real payment
provider reconciliation and a crash of the store itself are outside this model.
