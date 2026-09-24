// W2 验证专用 GameMode。UE 5.6.1

#include "GSWMVerifyGameMode.h"
#include "GSWMCaptureActor.h"

AGSWMVerifyGameMode::AGSWMVerifyGameMode()
{
	// 验证阶段不需要玩家 Pawn
	DefaultPawnClass = nullptr;
}

void AGSWMVerifyGameMode::StartPlay()
{
	Super::StartPlay();

	UWorld* World = GetWorld();
	if (!World)
	{
		UE_LOG(LogTemp, Error, TEXT("[GSWM] GameMode: no world"));
		return;
	}

	AGSWMCaptureActor* Actor = World->SpawnActor<AGSWMCaptureActor>();
	UE_LOG(LogTemp, Warning, TEXT("[GSWM] GameMode spawned capture actor: %s"),
		Actor ? TEXT("OK") : TEXT("FAIL"));
}
