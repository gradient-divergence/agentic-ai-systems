"""Reference solution: only an owner can grant, within a UTC interval."""

from security_privacy_delegate_06 import Delegations as Starter
from security_privacy_delegate_06 import Grant

from refund_agent import contracts as c
from refund_agent.security.security_privacy_accounts import AccountRecords


class Delegations(Starter):
    def grant(self, grantor, subject, order_id, at, until):
        at, until = c._utc_time(at), c._utc_time(until)
        if not self.accounts.may_act(grantor, order_id):
            raise ValueError("only the owner may delegate")
        if until <= at or not subject or subject == grantor:
            raise ValueError("delegation needs a distinct subject and a positive interval")
        grant = Grant(grantor, subject, order_id, at, until)
        self.grants += (grant,)
        return grant

    def accounts_at(self, at):
        at = c._utc_time(at)
        return AccountRecords(
            tuple(
                sorted(
                    set(self.accounts.grants)
                    | {
                        (g.subject, g.order_id)
                        for g in self.grants
                        if g.granted_at <= at < g.expires_at
                    }
                )
            )
        )
