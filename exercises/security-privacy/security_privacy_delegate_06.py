"""Exercise 6 starter: deliberately permanent, transferable delegated access."""

from dataclasses import dataclass
from datetime import datetime

from refund_agent.security.security_privacy_accounts import AccountRecords


@dataclass(frozen=True)
class Grant:
    grantor: str
    subject: str
    order_id: str
    granted_at: datetime
    expires_at: datetime


class Delegations:
    def __init__(self, accounts: AccountRecords):
        self.accounts = accounts
        self.grants: tuple[Grant, ...] = ()

    def grant(self, grantor, subject, order_id, at, until):
        if not self.accounts_at(at).may_act(grantor, order_id):
            raise ValueError("only the owner may delegate")
        grant = Grant(grantor, subject, order_id, at, until)
        self.grants += (grant,)
        return grant

    def accounts_at(self, at):
        return AccountRecords(
            tuple(
                sorted(
                    set(self.accounts.grants)
                    | {(g.subject, g.order_id) for g in self.grants if g.granted_at <= at}
                )
            )
        )
