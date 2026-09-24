// W2 最小验证用。UE 5.6.1
//
// 用法（命令行，无窗口）：
//   UnrealEditor-Cmd.exe <Project>.uproject <Map> -game -RenderOffScreen
//       -ResX=320 -ResY=240 -NoSplash -unattended -NullRHI=0
//       -GSWMOut=D:/gswm_out
//
// 注意：必须 -RenderOffScreen（有 RHI 无窗口），不能用 -nullrhi（完全没有渲染）。

#include "GSWMCaptureActor.h"

#include "ImageUtils.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/DirectionalLight.h"
#include "Components/StaticMeshComponent.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/CommandLine.h"
#include "HAL/PlatformApplicationMisc.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"
#include "TextureResource.h"

AGSWMCaptureActor::AGSWMCaptureActor()
{
	PrimaryActorTick.bCanEverTick = false;

	CaptureComp = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("CaptureComp"));
	RootComponent = CaptureComp;
}

void AGSWMCaptureActor::BeginPlay()
{
	Super::BeginPlay();

	// 允许命令行覆盖输出目录
	FString CmdOut;
	if (FParse::Value(FCommandLine::Get(), TEXT("-GSWMOut="), CmdOut))
	{
		OutputDir = CmdOut;
	}

	if (bSpawnTestScene)
	{
		SpawnTestScene();
	}

	SetupCapture();

	UE_LOG(LogTemp, Warning, TEXT("[GSWM] RenderOffScreen verify: %dx%d, %d frames -> %s"),
		CaptureWidth, CaptureHeight, NumFrames, *OutputDir);

	int32 Saved = 0;
	for (int32 i = 0; i < NumFrames; ++i)
	{
		TArray<FColor> Pixels;
		if (!GrabColorFrame(Pixels))
		{
			UE_LOG(LogTemp, Error, TEXT("[GSWM] frame %d: ReadPixels FAILED"), i);
			continue;
		}

		const FString Path = FPaths::Combine(
			OutputDir, FString::Printf(TEXT("frame_%03d.png"), i));

		if (SavePNG(Path, CaptureWidth, CaptureHeight, Pixels))
		{
			++Saved;
			UE_LOG(LogTemp, Warning, TEXT("[GSWM] saved %s (%d pixels)"), *Path, Pixels.Num());
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("[GSWM] DONE: %d/%d frames saved"), Saved, NumFrames);

	// 验证阶段：采完直接退出，不等用户操作
	FinishAndQuit(Saved > 0 ? 0 : 1);
}

void AGSWMCaptureActor::SetupCapture()
{
	// 颜色 RT（阶段 1：只验证 RGB）
	RT_Color = NewObject<UTextureRenderTarget2D>(this);
	RT_Color->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
	RT_Color->InitAutoFormat(CaptureWidth, CaptureHeight);
	RT_Color->bAutoGenerateMips = false;
	RT_Color->UpdateResourceImmediate(true);

	CaptureComp->TextureTarget = RT_Color;
	CaptureComp->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;  // LDR RGB
	CaptureComp->bCaptureEveryFrame = false;   // 手动 CaptureScene()
	CaptureComp->bCaptureOnMovement = false;
	CaptureComp->bAlwaysPersistRenderingState = true;  // 离屏下保持渲染状态
	CaptureComp->FOVAngle = 60.f;

	// 可选：G-buffer RT（RGBA16F，配合后处理材质一次输出 normal.xy/depth/stencil）
	if (bExportGBuffer)
	{
		RT_GBuffer = NewObject<UTextureRenderTarget2D>(this);
		RT_GBuffer->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA16f;
		RT_GBuffer->InitAutoFormat(CaptureWidth, CaptureHeight);
		RT_GBuffer->bAutoGenerateMips = false;
		RT_GBuffer->UpdateResourceImmediate(true);
		// 用法：给 CaptureComp 挂后处理材质，材质内采样
		//   SceneTexture:WorldNormal / SceneTexture:SceneDepth / SceneTexture:CustomStencil
		// 输出到 RGBA16F = (normal.x, normal.y, depth, stencil)
		// 读回必须用 ReadPixels(TArray<FFloat16Color>&) 版本，RGBA8 会毁掉精度。
	}
}

bool AGSWMCaptureActor::GrabColorFrame(TArray<FColor>& OutPixels)
{
	if (!CaptureComp || !RT_Color) return false;

	CaptureComp->CaptureScene();   // 立即渲染一帧到 RT

	FTextureRenderTargetResource* Rsrc = RT_Color->GameThread_GetRenderTargetResource();
	if (!Rsrc) return false;

	OutPixels.Reset();
	// 同步读回：慢，但验证阶段够用。后面再换 FRHIGPUTextureReadback 异步。
	return Rsrc->ReadPixels(OutPixels);
}

void AGSWMCaptureActor::SpawnTestScene()
{
	UWorld* World = GetWorld();
	if (!World) return;

	// 立方体：引擎自带基础形状，放大到画面里能看清
	UStaticMesh* CubeMesh =
		LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	if (CubeMesh)
	{
		AStaticMeshActor* Cube = World->SpawnActor<AStaticMeshActor>();
		if (Cube && Cube->GetStaticMeshComponent())
		{
			Cube->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
			Cube->SetActorLocation(FVector::ZeroVector);
			Cube->SetActorScale3D(FVector(2.f));
			Cube->SetActorLabel(TEXT("GSWM_TestCube"));
			UE_LOG(LogTemp, Warning, TEXT("[GSWM] spawned test cube"));
		}
	}

	// 平行光：没有光就是全黑
	ADirectionalLight* Light = World->SpawnActor<ADirectionalLight>();
	if (Light)
	{
		Light->SetActorRotation(FRotator(-50.f, 30.f, 0.f));
		Light->SetActorLabel(TEXT("GSWM_TestLight"));
	}

	// 相机放到立方体斜前方并朝向它
	const FVector CamPos(0.f, -400.f, 150.f);
	SetActorLocation(CamPos);
	SetActorRotation((FVector::ZeroVector - CamPos).GetSafeNormal().Rotation());
}

bool AGSWMCaptureActor::SavePNG(const FString& Path, int32 W, int32 H, const TArray<FColor>& Pixels)
{
	IFileManager::Get().MakeDirectory(*FPaths::GetPath(Path), true);

	// 用 Engine 自带的 FImageUtils，省得给 Build.cs 加 ImageWrapper 依赖
	TArray<uint8> Compressed;
	FImageUtils::CompressImageArray(W, H, Pixels, Compressed);
	if (Compressed.Num() == 0)
	{
		return false;
	}
	return FFileHelper::SaveArrayToFile(Compressed, *Path);
}

void AGSWMCaptureActor::FinishAndQuit(int32 ReturnCode)
{
	UE_LOG(LogTemp, Warning, TEXT("[GSWM] quitting with code %d"), ReturnCode);
	FPlatformMisc::RequestExit(false);
}
