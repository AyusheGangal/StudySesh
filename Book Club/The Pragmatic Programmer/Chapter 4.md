You can't write perfect software.
	- opens the chapter and gets straight to the point.
	- Asks us to embrace it.
	- Perfect software does not exist.

The chapter is about how pragmatic programmer turns this depressing yet true point into an advantage.

- Analogy is drawn between "us being the best driver on the earth, and so we drive defensively", and "we also code defensively by writing assertions, check for consistency, put constraints on the database cols".
	- Pragmatic programmers don't even trust themselves, and build defenses against their own mistakes!

Index:
- First such defensive measure is talked about in Design by contract where clients and suppliers must agree on rights and responsibilities

### Design by Contract
- A contract is any document that defines your rights and responsibilities, and also of the other involved party. It also has an agreement concerning repercussions if either party fails to deliver/ abide by the contract.
- Software module interaction/ design should have a contract
- Popularized and developed by Bertrand Meyer for the language "Eiffel"
- It focuses on documenting and agreeing to the rights and responsibilities of software modules to ensure program correctness.
	- A correct program is one which does no more and no less than it claims to do.
- Has 3 main components:
	- Pre conditions: must be true in order for the routine/method/function to be called
	- Post condition: What will be true/ or is guaranteed to do when the routine/method/function is called
		- It will conclude
		- infinite loops aren't allowed
	- Class invariant: A condition that is/will always be true from the perspective of the caller before the routine is called and after the routine is called. May or may not hold true while the routine is in process.
- By def: If all the routine's preconditions are met by the caller, the routine shall guarantee that all the postconditions and invariants will be true when it completes.
- If for some reason either party fails, a remedy is invoked
	- An exception
	- program terminates
	- It is a bug, it should never happen
- Real world example:
	- Real-world Analogy: ATM Withdrawal
		- Imagine using an ATM to withdraw money. Here's how **Design by Contract** applies:
		- **Precondition**:  You must have enough money in your account. Your card and PIN must be valid.
		- **Postcondition**:  You get the exact amount of cash requested. Your account balance is reduced by that amount.
		- **Invariant**:  The bank account must always have a non-negative balance. You can’t go below zero.
		The ATM trusts you to follow the precondition (having enough money), and in return, it follows through on the postcondition (dispenses the cash). If either side breaks the contract, something fails (you get a warning, or the bank gets sued).

```Python
class BankAccount:
    def __init__(self, balance):
        assert balance >= 0, "Initial balance must be non-negative"  
        # Invariant
        self.balance = balance

    def deposit(self, amount):
        assert amount > 0, "Deposit amount must be positive"  # Precondition
        old_balance = self.balance
        self.balance += amount
        assert self.balance == old_balance + amount  # Postcondition

    def withdraw(self, amount):
        assert 0 < amount <= self.balance, "Invalid withdraw amount"  
        # Precondition
        old_balance = self.balance
        self.balance -= amount
        assert self.balance == old_balance - amount  # Postcondition

    def get_balance(self):
        assert self.balance >= 0, "Balance must always be non-negative" 
        # Invariant
        return self.balance
```

- write lazy code: be strict about what you accept before you begin, and promise as little as possible. -- avoid mistakes
- Class invariants: is basically the state. you pass a state, get updated state.
- Comparison of DbC and TDD/Testing:
	- -- see paragraph--
- Implementing DbC:
	- Document the assumptions, range of input domain, boundary conditions, what method will do, what it does not do
	- Use assertions for the compiler to check all your documented conditions (pre, post, state) partially emulate

#### DbC and crashing early:
By using preconditions, post conditions and invariants, you can crash early and get more accurate information about the problem.
