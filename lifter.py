import struct

from typing import List
from .instruction import GameBoyInstruction
from binaryninja.architecture import Architecture
from binaryninja import LowLevelILLabel, LowLevelILOperation

def u8(bs, signed=False):
    x, = struct.unpack('<' + 'b' if signed else 'B', bs)
    return x

def u16(bs, signed=False):
    x, = struct.unpack("<" + 'h' if signed else 'H', bs)
    return x

def cond_branch_concrete(il, cond, true_addr, false_addr):
    t = il.get_label_for_address(
        Architecture['LR35902'],
        true_addr
    )

    if t is None:
        # t is not an address in the current function scope.
        t = LowLevelILLabel()
        indirect = True
    else:
        indirect = False

    f_label_found = True

    f = il.get_label_for_address(
        Architecture['LR35902'],
        false_addr
    )

    if f is None:
        f = LowLevelILLabel()
        f_label_found = False

    il.append(il.if_expr(cond, t, f))

    if indirect:
        # If the destination is not in the current function,
        # then a jump, rather than a goto, needs to be added to
        # the IL.
        il.mark_label(t)
        il.append(il.jump(il.const_pointer(2, true_addr)))

    if not f_label_found:
        il.mark_label(f)


def jump_concrete(il, dest_addr):
    label = il.get_label_for_address(
        Architecture['LR35902'],
        dest_addr
    )

    if label is None:
        return il.jump(il.const_pointer(2, dest_addr))
    else:
        return il.goto(label)

REGS_8 = {'A', 'B', 'C', 'D', 'E', 'F', 'H', 'L'}
REGS_16 = {'AF', 'BC', 'DE', 'HL', 'SP', 'PC'}
class GameBoyLifter:
    def __init__(self, il_function) -> None:
        self.il = il_function

    def _lift_operand_expr(self, insn: GameBoyInstruction, op: str, size=None):
        data = insn.data
        if op[0] == '(':
            assert op[-1] == ')', op
            assert size is not None
            addr_expr, addr_size = self._lift_operand_expr(insn, op[1:-1], size=2)
            assert addr_size == 2
            return self.il.load(size, addr_expr), size

        elif op == "C_IO":
            return self.il.add(
                2,
                self.il.zero_extend(
                    2,
                    self.il.reg(1, 'C')),
                    self.il.const(2, 0xff00)
            ), 2

        elif op == 'd8':
            return self.il.const(1, u8(data[1:2], signed=True)), 1

        elif op == 'offset8':
            return self.il.sign_extend(2, self.il.const(1, u8(data[1:2], signed=True))), 2

        elif op == 'd16':
            return self.il.const(2, u16(data[1:3], signed=True)), 2

        elif op == 'a8':
            return self.il.const_pointer(2, 0xff00 + u8(data[1:2])), 2

        elif op == 'r8':
            return self.il.const_pointer(2, (insn.addr + insn.length + u8(data[1:2], signed=True)) & 0xffff), 2

        elif op == 'a16':
            return self.il.const_pointer(2, u16(data[1:3], signed=False)), 2

        elif op in {'00H', '08H', '10H', '18H', '20H', '28H', '30H', '38H'}:
            return self.il.const_pointer(2, int(op[:-1], base=16)), 2

        elif op in REGS_8:
            return self.il.reg(1, op), 1

        elif op in REGS_16:
            return self.il.reg(2, op), 2

        elif op == 'SP+d8':
            return self.il.add(2, self.il.reg(2, 'SP'), self.il.const(2, u8(data[1:2], signed=True))), 2

        else:
            assert False, f"WARNING: Unimplemented operand pattern: {insn=} {op=}"

    def _lift_arith_one(self, insn: GameBoyInstruction):
        operation = {
            'OR': lambda old, new: self.il.or_expr(1, old, new, flags="*"),
            'XOR': lambda old, new: self.il.xor_expr(1, old, new, flags="*"),
            'AND': lambda old, new: self.il.and_expr(1, old, new, flags="*"),
            'SUB': lambda old, new: self.il.sub(1, old, new, flags="*"),
            'CP': lambda old, new: self.il.add(1, self.il.sub(1, old, new, flags="*"), new)
        }
        src, = insn.operand_symbols
        new_expr, src_size = self._lift_operand_expr(insn, src, size=1)
        assert src_size == 1, f"{insn=}"
        old_expr =  self.il.reg(1, 'A')

        result_expr = operation[insn.mnemonic](old_expr, new_expr)
        self.il.append(self.il.set_reg(1, 'A', result_expr))
        return True

    def _lift_instruction_ADD(self, insn: GameBoyInstruction):
        dst, src = insn.operand_symbols
        if dst == 'A':
            src_expr, src_size = self._lift_operand_expr(insn, src, size=1)
            assert src_size == 1, f"expected src_size == 1, got {src_size=} for {insn=}"
            self.il.append(self.il.set_reg(1, 'A', self.il.add(1, self.il.reg(1, 'A'), src_expr, flags="*")))
            return True
        elif dst in {'SP', 'HL'}:
            src_expr, src_size = self._lift_operand_expr(insn, src, size=2)
            assert src_size == 2, f"expected src_size == 2, got {src_size=} for {insn=}"
            self.il.append(self.il.set_reg(2, dst, self.il.add(2, self.il.reg(2, dst), src_expr, flags="*")))
            return True

    def _lift_arith_two(self, insn: GameBoyInstruction):
        dst, src = insn.operand_symbols
        assert dst == 'A'
        reg_A = self.il.reg(1, 'A')
        src_expr, src_size = self._lift_operand_expr(insn, src, size=1)
        assert src_size == 1, f"expected src_size == 1, got {src_size=} for {insn=}"

        operation = {
            'SBC': lambda:  self.il.sub_borrow(1, reg_A, src_expr, self.il.flag('c'), flags='*'),
            'ADC': lambda: self.il.add_carry(1, reg_A, src_expr, self.il.flag('c'), flags='*'),
        }[insn.mnemonic]
        self.il.append(self.il.set_reg(1, 'A', operation()))
        return True

    def _lift_instruction_INC(self, insn: GameBoyInstruction):
        reg, = insn.operand_symbols
        if reg == '(HL)':
            # fucking special cases
            src_expr, src_size = self._lift_operand_expr(insn, reg, 1)
            assert src_size == 1
            new_value = self.il.add(1, src_expr, self.il.const(1, 1), flags="*")
            self.il.append(self.il.store(1, self.il.reg(2, 'HL'), new_value))
            return True

        size = 1 if reg in REGS_8 else 2
        assert size == 1 or reg in {'BC', 'DE', 'HL', 'SP'}, f"{insn=}"
        self.il.append(self.il.set_reg(size, reg, self.il.add(size, self.il.reg(size, reg), self.il.const(size, 1), flags="*")))
        return True

    def _lift_instruction_DEC(self, insn: GameBoyInstruction):
        reg, = insn.operand_symbols
        if reg == '(HL)':
            # fucking special cases
            src_expr, src_size = self._lift_operand_expr(insn, reg, 1)
            assert src_size == 1
            new_value = self.il.sub(1, src_expr, self.il.const(1, 1), flags="*")
            self.il.append(self.il.store(1, self.il.reg(2, 'HL'), new_value))
            return True
        size = 1 if reg in REGS_8 else 2
        assert size == 1 or reg in {'BC', 'DE', 'HL', 'SP'}, f"{insn=}"
        self.il.append(self.il.set_reg(size, reg, self.il.sub(size, self.il.reg(size, reg), self.il.const(size, 1), flags="*")))
        return True

    def _lift_instruction_CPL(self, insn: GameBoyInstruction):
        self.il.append(self.il.set_reg(1, 'A', self.il.not_expr(1, self.il.reg(1, 'A'))))
        return True

    def _lift_instruction_LDH(self, insn: GameBoyInstruction):
        addr = self.il.const_pointer(2, 0xff00 + u8(insn.data[1:2]))
        if insn.operand_symbols[0][0] == '(':
            # LDH (a8), A == store
            self.il.append(self.il.store(1, addr, self.il.reg(1, 'A')))
        else:
            # LDH A, (a8)
            self.il.append(self.il.set_reg(1, 'A', self.il.load(1, addr)))
        return True

    def _lift_instruction_BIT(self, insn: GameBoyInstruction):
        bit_idx, src_sym = insn.operand_symbols
        src_expr, src_size = self._lift_operand_expr(insn, src_sym, size=1)
        assert src_size == 1
        assert bit_idx in set('01234567')
        bit_idx = int(bit_idx)

        self.il.append(self.il.set_flag(
            'z',
            self.il.compare_equal(
                0,
                self.il.const(1, 0),
                self.il.and_expr(
                    1,
                    src_expr,
                    self.il.const(1, 1<<bit_idx)
                )
            )
        ))
        return True

    def _lift_instruction_RES(self, insn: GameBoyInstruction):
        bit_idx, src_sym = insn.operand_symbols
        src_expr, src_size = self._lift_operand_expr(insn, src_sym, size=1)
        assert src_size == 1
        assert bit_idx in set('01234567')
        bit_idx = int(bit_idx)
        mask = self.il.const(1, 0xff - (1 << bit_idx))
        new_val = self.il.and_expr(1, src_expr, mask)

        if src_sym == '(HL)':
            self.il.append(self.il.store(1, self.il.reg(2, 'HL'), new_val))
        else:
            assert src_sym in REGS_8
            self.il.append(self.il.set_reg(1, src_sym, new_val))
        return True

    def _lift_shifts(self, insn: GameBoyInstruction):
        src_sym, = insn.operand_symbols
        src_expr, src_size = self._lift_operand_expr(insn, src_sym, size=1)
        assert src_size == 1

        _C = self.il.flag('c')
        _1 = self.il.const(1, 1)

        op = {
            'RLC': lambda: self.il.rotate_left(1, src_expr, _1, flags='*'),
            'RRC': lambda: self.il.rotate_right(1, src_expr, _1, flags='*'),
            'RL': lambda: self.il.rotate_left_carry(1, src_expr, _1, _C, flags='*'),
            'RR': lambda: self.il.rotate_right_carry(1, src_expr, _1, _C, flags='*'),
            'SLA': lambda: self.il.shift_left(1, src_expr, _1, flags='*'),
            'SRA': lambda: self.il.arith_shift_right(1, src_expr, _1, flags='*'),
            'SRL': lambda: self.il.logical_shift_right(1, src_expr, _1, flags='*')
        }[insn.mnemonic]
        shifted = op()
        if src_sym in REGS_8:
            self.il.append(self.il.set_reg(1, src_sym, shifted))
        elif src_sym == '(HL)':
            self.il.append(self.il.store(1, self.il.reg(2, 'HL'), shifted))
        else:
            assert False, f"Invalid operand for {insn=}"
        return True

    def _lift_accumulator_rotates(self, insn: GameBoyInstruction):
        assert not insn.operand_symbols

        _A = self.il.reg(1, 'A')
        _C = self.il.flag('c')
        _1 = self.il.const(1, 1)

        op = {
            'RLCA': lambda: self.il.rotate_left(1, _A, _1, flags='*'),
            'RLA': lambda: self.il.rotate_left_carry(1, _A, _1, _C, flags='*'),
            'RRCA': lambda: self.il.rotate_right(1, _A, _1, flags='*'),
            'RRA': lambda: self.il.rotate_right_carry(1, _A, _1, _C, flags='*'),
        }[insn.mnemonic]
        self.il.append(self.il.set_reg(1, 'A', op()))
        return True

    def _lift_instruction_LD(self, insn: GameBoyInstruction):
        dst, src = insn.operand_symbols
        if dst[0] == '(': # dst is deref == store
            assert dst[-1] == ')'
            dst_addr_symbol = dst[1:-1]
            update = None
            if dst_addr_symbol[-1] in '+-':
                dst_addr_symbol, update = dst_addr_symbol[:-1], dst_addr_symbol[-1]

            src_expr, src_size = self._lift_operand_expr(insn, src) # lift the source so we know the size of the store
            addr_expr, addr_size = self._lift_operand_expr(insn, dst_addr_symbol)
            assert addr_size == 2, f"expected the size of {dst_addr_symbol=} to be 2, got {addr_expr=}, {addr_size=} for {insn=}"

            self.il.append(self.il.store(src_size, addr_expr, src_expr))
            if update:
                assert dst_addr_symbol == 'HL'
                op = {'-': self.il.sub, '+': self.il.add}[update]
                new_val = op(2, addr_expr, self.il.const(addr_size, 1))

                self.il.append(self.il.set_reg(2, 'HL', new_val))
            return True

        elif src[0] == '(': # src is deref == load
            assert src[-1] == ')'
            src_addr_symbol = src[1:-1]
            update = None
            if src_addr_symbol[-1] in '+-':
                src_addr_symbol, update = src_addr_symbol[:-1], src_addr_symbol[-1]

            assert dst in {'A', 'B', 'C', 'D', 'E', 'H', 'L'}, f"memory loads are only allowed into single byte registers! {insn=}"
            addr_expr, addr_size = self._lift_operand_expr(insn, src_addr_symbol)
            assert addr_size == 2, f"expected the size of {src_addr_symbol=} to be 2, got {addr_expr=}, {addr_size=} for {insn=}"


            self.il.append(self.il.set_reg(1, dst, self.il.load(1, addr_expr)))
            if update:
                assert src_addr_symbol == 'HL'
                assert dst == 'A'
                op = {'-': self.il.sub, '+': self.il.add}[update]
                new_val = op(2, addr_expr, self.il.const(addr_size, 1))

                self.il.append(self.il.set_reg(2, 'HL', new_val))
            return True

        elif dst in REGS_8:
            src_expr, src_size = self._lift_operand_expr(insn, src)
            assert src_size == 1, f"load into 1byte register is not 1 byte?? {insn=} {src_size=}"
            self.il.append(self.il.set_reg(1, dst, src_expr))
            return insn.length

        elif dst in REGS_16:
            src_expr, src_size = self._lift_operand_expr(insn, src)
            assert src_size == 2, f"load into 2byte register is not 2 bytes?? {insn=} {src_size=}"
            self.il.append(self.il.set_reg(2, dst, src_expr))
            return True

        print(f"Unhandled LD instruction mode: {insn=}")
        return False

    def _lift_instruction_JR(self, insn: GameBoyInstruction):
        off_dest = u8(insn.data[1:2], signed=True)
        # print(insn)

        if len(insn.operand_symbols) == 1:
            self.il.append(jump_concrete(self.il, insn.addr + insn.length + off_dest))
        else:
            flagcond, _ = insn.operand_symbols
            needed = self.il.const(0, 1 if flagcond[0] != 'N' else 0)
            flag = self.il.flag(flagcond[-1].lower())
            cond_branch_concrete(
                self.il,
                self.il.compare_equal(0, flag, needed),
                insn.addr + insn.length + off_dest,
                insn.addr + insn.length
            )
        return True

    def _lift_instruction_JP(self, insn: GameBoyInstruction):
        if insn.operand_symbols == ['HL']:
            self.il.append(self.il.jump(self.il.reg(2, 'HL')))
            return True
        assert len(insn.data) >= 3, f"FUCK {insn=}"
        addr = u16(insn.data[1:3])
        # print(insn)

        if len(insn.operand_symbols) == 1:
            self.il.append(jump_concrete(self.il, addr))
        else:
            flagcond, _ = insn.operand_symbols
            needed = self.il.const(0, 1 if flagcond[0] != 'N' else 0)
            flag = self.il.flag(flagcond[-1].lower())
            cond_branch_concrete(
                self.il,
                self.il.compare_equal(0, flag, needed),
                addr,
                insn.addr + insn.length
            )
        return True

    def _lift_instruction_RET(self, insn: GameBoyInstruction):
        if not insn.operand_symbols:
            # unconditional ret
            sp = self.il.reg(2, 'SP')
            _2 = self.il.const(2, 2)
            self._add_reg('SP', self.il.const(2, 2), size=2)

            self.il.append(self.il.ret(
                self.il.load(
                    2,
                    self.il.sub(2, self.il.reg(2, 'SP'), _2)
                )))
            return True

        cond, = insn.operand_symbols
        negated = cond[0] == 'N'
        flag = cond[-1]

        # not sure yet, deal with it later
        return False

    def _lift_instruction_RETI(self, insn: GameBoyInstruction):
        assert not insn.operand_symbols
            # unconditional ret
        sp = self.il.reg(2, 'SP')
        _2 = self.il.const(2, 2)
        self._add_reg('SP', self.il.const(2, 2), size=2)

        self.il.append(self.il.ret(
            self.il.load(
                2,
                self.il.sub(2, self.il.reg(2, 'SP'), _2)
            )))
        return True

    def _lift_instruction_EI(self, insn: GameBoyInstruction):
        self.il.append(self.il.nop())
        return True

    def _lift_instruction_DI(self, insn: GameBoyInstruction):
        self.il.append(self.il.nop())
        return True

    def _lift_instruction_SCF(self, insn: GameBoyInstruction):
        self.il.append(self.il.set_flag('c', self.il.const(0, 1)))
        self.il.append(self.il.set_flag('n', self.il.const(0, 0)))
        self.il.append(self.il.set_flag('h', self.il.const(0, 0)))
        return True

    def _lift_instruction_CCF(self, insn: GameBoyInstruction):
        self.il.append(self.il.set_flag('h', self.il.flag('c')))
        self.il.append(self.il.set_flag('n', self.il.const(0, 0)))
        self.il.append(self.il.set_flag('c', self.il.not_expr(0, self.il.flag('c'))))
        return True

    def _lift_instruction_CALL(self, insn: GameBoyInstruction):
        if len(insn.operand_symbols) == 1:
            # unconditional CALL
            addr_sym, = insn.operand_symbols
            addr_expr, addr_size = self._lift_operand_expr(insn, addr_sym, size=2)
            assert addr_size == 2
            self.il.append(self.il.call(addr_expr))
            return True

        cond, addr_sym = insn.operand_symbols
        negated = cond[0] == 'N'
        flag = cond[-1]

        # not sure yet, deal with it later
        return False

    def _lift_instruction_RST(self, insn: GameBoyInstruction):
        dst, = insn.operand_symbols
        assert dst.endswith('H')
        self.il.append(self.il.call(self.il.const(2, int(dst[:-1], base=16))))
        return True

    def _lift_instruction_NOP(self, insn: GameBoyInstruction):
        assert not insn.operand_symbols, f"{insn=}"

        self.il.append(self.il.nop())
        return True

    def _lift_instruction_SWAP(self, insn: GameBoyInstruction):
        op, = insn.operand_symbols

        arg_expr, arg_size = self._lift_operand_expr(insn, op, size=1)
        assert arg_size == 1
        _4 = self.il.const(1, 4)
        _0xf = self.il.const(1, 0xf)
        new_high = self.il.shift_left(1, self.il.and_expr(1, arg_expr, _0xf), _4)
        new_low = self.il.and_expr(1, self.il.logical_shift_right(1, arg_expr, _4), _0xf)
        result = self.il.or_expr(1, new_high, new_low)
        if op in REGS_8:
            self.il.append(self.il.set_reg(1, op, result))
        elif op == '(HL)':
            self.il.append(self.il.store(1, self.il.reg(2, "HL"), result))
        else:
            assert False, f"Incorrect operand for SWAP: {op=} in {insn=}"
        return True

    def _update_reg(self, reg, rhs, op, size=2):
        assert (size == 2 and reg in REGS_16) or (size == 1 and reg in REGS_8)
        self.il.append(
            self.il.set_reg(
                size,
                reg,
                op(
                    size,
                    self.il.reg(
                        size,
                        reg
                    ),
                    rhs
                )
            )
        )

    def _add_reg(self, reg, rhs, size=2):
        self._update_reg(reg, rhs, self.il.add, size=size)

    def _sub_reg(self, reg, rhs, size=2):
        self._update_reg(reg, rhs, self.il.sub, size=size)

    def _lift_instruction_POP(self, insn):
        reg, = insn.operand_symbols
        assert reg in {'AF', 'BC', 'DE', 'HL'}

        loaded_val = self.il.load(2, self.il.reg(2, 'SP'))
        _2 = self.il.const(2, 2)

        self.il.append(self.il.set_reg(2, reg, loaded_val))
        self._add_reg('SP', _2, size=2)
        return True

    def _lift_instruction_PUSH(self, insn):
        reg, = insn.operand_symbols
        assert reg in {'AF', 'BC', 'DE', 'HL'}

        self._sub_reg('SP', self.il.const(2, 2), size=2)
        self.il.append(self.il.store(2, self.il.reg(2, 'SP'), self.il.reg(2, reg)))

        return True

    def _lift_instruction_HALT(self, insn):
        self.il.append(self.il.no_ret())
        return True

    def lift(self, insn: GameBoyInstruction):
        f = None
        if insn.mnemonic in {'AND', 'OR', 'XOR', 'SUB', 'CP'}:
            f = self._lift_arith_one
        elif insn.mnemonic in {'SBC', 'ADC'}:
            f = self._lift_arith_two
        elif insn.mnemonic in { 'RLC', 'RRC', 'RL', 'RR', 'SLA', 'SRA', 'SRL'}:
            f = self._lift_shifts
        elif insn.mnemonic in { 'RLCA', 'RLA', 'RRCA', 'RRA'}:
            f = self._lift_accumulator_rotates
        # print(insn)
        # this is janky so we can notice duplicate implementations, can probably be removed in the future
        try:
            f2 = getattr(self, f'_lift_instruction_{insn.mnemonic}')
            assert f is None, f"duplicate! unnecessary: {f2=}"
            f = f2
        except AttributeError:
            if f is None:
                print(f"Unhandled instruction type, cannot lift {insn=}")
            return None

        result = f(insn)
        # if not result:
        #     print(f"Failed to lift LLIL for {insn=}")

        return insn.length if result else None