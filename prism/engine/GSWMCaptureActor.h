// W2 最小验证用：离屏渲染 + SceneCapture 读回 + 存 PNG。
// 目标只有一个——证明 "-RenderOffScreen 下能拿到画面"，不做任何桥接。
// UE 5.6.1

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GSWMCaptureActor.generated.h"

UCLASS()
class AGSWMCaptureActor : public AActor
{
	GENERATED_BODY()

public:
	AGSWMCaptureActor();

	/** 采集分辨率 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GSWM")
	int32 CaptureWidth = 64;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GSWM")
	int32 CaptureHeight = 64;

	/** 采集帧数（验证阶段 8 帧足够） */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GSWM")
	int32 NumFrames = 8;

	/** 输出目录，可用命令行覆盖: -GSWMOut=D:/xxx */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GSWM")
	FString OutputDir = TEXT("Saved/GSWM");

	/** 是否同时导出法线+深度（后处理材质路线，W2 阶段 2 再开） */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GSWM")
	bool bExportGBuffer = false;

	/** 自动生成最小测试场景（Cube + 平行光），这样关卡可以是空的，无需手动摆放 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "GSWM")
	bool bSpawnTestScene = true;

protected:
	virtual void BeginPlay() override;

private:
	UPROPERTY()
	TObjectPtr<USceneCaptureComponent2D> CaptureComp;

	UPROPERTY()
	TObjectPtr<UTextureRenderTarget2D> RT_Color;

	UPROPERTY()
	TObjectPtr<UTextureRenderTarget2D> RT_GBuffer;

	void SetupCapture();
	void SpawnTestScene();
	bool GrabColorFrame(TArray<FColor>& OutPixels);
	bool SavePNG(const FString& Path, int32 W, int32 H, const TArray<FColor>& Pixels);
	void FinishAndQuit(int32 ReturnCode);
};
