; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%d \00"
@input_matrix = private global [2 x [16 x i8]] [[16 x i8] c"\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F", [16 x i8] c"\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F"]

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @printf(ptr, ...)

define i8 @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i8, ptr null, i64 32) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 2, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 16, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 16, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %24, %0
  %10 = phi i64 [ %26, %24 ], [ 0, %0 ]
  %11 = icmp slt i64 %10, 2
  br i1 %11, label %12, label %27

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %23, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 16
  br i1 %15, label %16, label %24

16:                                               ; preds = %13
  %17 = mul i64 %10, 16
  %18 = add i64 %17, %14
  %19 = getelementptr i8, ptr @input_matrix, i64 %18
  %20 = load i8, ptr %19, align 1
  %21 = sext i8 %20 to i32
  %22 = call i32 (ptr, ...) @printf(ptr @frmt_spec, i32 %21)
  %23 = add i64 %14, 1
  br label %13

24:                                               ; preds = %13
  %25 = call i32 (ptr, ...) @printf(ptr @nl)
  %26 = add i64 %10, 1
  br label %9

27:                                               ; preds = %9
  br label %28

28:                                               ; preds = %44, %27
  %29 = phi i64 [ %46, %44 ], [ 0, %27 ]
  %30 = icmp slt i64 %29, 2
  br i1 %30, label %31, label %47

31:                                               ; preds = %28
  br label %32

32:                                               ; preds = %35, %31
  %33 = phi i64 [ %43, %35 ], [ 0, %31 ]
  %34 = icmp slt i64 %33, 16
  br i1 %34, label %35, label %44

35:                                               ; preds = %32
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %37 = mul i64 %29, 16
  %38 = add i64 %37, %33
  %39 = getelementptr i8, ptr %36, i64 %38
  %40 = load i8, ptr %39, align 1
  %41 = sext i8 %40 to i32
  %42 = call i32 (ptr, ...) @printf(ptr @frmt_spec, i32 %41)
  %43 = add i64 %33, 1
  br label %32

44:                                               ; preds = %32
  %45 = call i32 (ptr, ...) @printf(ptr @nl)
  %46 = add i64 %29, 1
  br label %28

47:                                               ; preds = %28
  call void @llvm.riscv.bb.mvin(i64 ptrtoint (ptr @input_matrix to i64), i64 4294967306)
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %49 = ptrtoint ptr %48 to i64
  call void @llvm.riscv.bb.mvout(i64 %49, i64 537133066)
  br label %50

50:                                               ; preds = %66, %47
  %51 = phi i64 [ %68, %66 ], [ 0, %47 ]
  %52 = icmp slt i64 %51, 2
  br i1 %52, label %53, label %69

53:                                               ; preds = %50
  br label %54

54:                                               ; preds = %57, %53
  %55 = phi i64 [ %65, %57 ], [ 0, %53 ]
  %56 = icmp slt i64 %55, 16
  br i1 %56, label %57, label %66

57:                                               ; preds = %54
  %58 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %59 = mul i64 %51, 16
  %60 = add i64 %59, %55
  %61 = getelementptr i8, ptr %58, i64 %60
  %62 = load i8, ptr %61, align 1
  %63 = sext i8 %62 to i32
  %64 = call i32 (ptr, ...) @printf(ptr @frmt_spec, i32 %63)
  %65 = add i64 %55, 1
  br label %54

66:                                               ; preds = %54
  %67 = call i32 (ptr, ...) @printf(ptr @nl)
  %68 = add i64 %51, 1
  br label %50

69:                                               ; preds = %50
  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  call void @free(ptr %70)
  ret i8 0
}

; Function Attrs: nounwind
declare void @llvm.riscv.bb.mvin(i64, i64) #0

; Function Attrs: nounwind
declare void @llvm.riscv.bb.mvout(i64, i64) #0

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
