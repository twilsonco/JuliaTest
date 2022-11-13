using Yao
using FFTW, LinearAlgebra

# https://docs.yaoquantum.org/v0.5/examples/QFT/

A(i, j) = control(i, j=>shift(2π/(1<<(i-j+1))))

R4 = A(4, 1)
(n -> control(n, 4, 1 => shift(0.392699)))

# mat(R4(5))

B(n, k) = chain(n, j==k ? put(k=>H) : A(j, k) for j in k:n)

qft(n) = chain(B(n, k) for k in 1:n)

# qft(4)

struct QFT{N} <: PrimitiveBlock{N} end
QFT(n::Int) = QFT{n}()

circuit(::QFT{N}) where N = qft(N)

YaoBlocks.mat(::Type{T}, x::QFT) where T = mat(T, circuit(x))

YaoBlocks.print_block(io::IO, x::QFT{N}) where N = print(io, "QFT($N)")


function YaoBlocks.apply!(r::ArrayReg, x::QFT)
    α = sqrt(length(statevec(r)))
    invorder!(r)
    lmul!(α, ifft!(statevec(r)))
    return r
end

# r = rand_state(5)
# r1 = r |> copy |> QFT(5)
# r2 = r |> copy |> circuit(QFT(5))
# r1 ≈ r2

# QFT(5)'

# Phase Estimation

Hadamards(n) = repeat(H, 1:n)

ControlU(n, m, U) = chain(n+m, control(k, n+1:n+m=>matblock(U^(2^(k-1)))) for k in 1:n)

PE(n, m, U) =
    chain(n+m, # total number of the qubits
        concentrate(Hadamards(n), 1:n), # apply H in local scope
        ControlU(n, m, U),
        concentrate(QFT(n)', 1:n))

r = rand_state(5)

focus!(r, 1:3)

relax!(r, 1:3)

# N, M = 3, 5
# P = eigen(rand_unitary(1<<M)).vectors
# θ = Int(0b110) / 1<<N
# phases = rand(1<<M)
# phases[bit"010"] = θ