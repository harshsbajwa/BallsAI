import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Stack } from "expo-router";
import { useState } from "react";
import "../global.css";
import { useColorScheme } from "react-native";
import { NAV_THEME } from "@/lib/constants";


export default function RootLayout() {
  const [queryClient] = useState(() => new QueryClient());
  const colorScheme = useColorScheme();

  return (
    <QueryClientProvider client={queryClient}>
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: NAV_THEME[colorScheme ?? "light"].background,
          },
          headerTintColor: NAV_THEME[colorScheme ?? "light"].text,
          headerTitleStyle: {
            fontWeight: "bold",
          },
        }}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="player/[id]"
          options={{ title: "Player Details" }}
        />
        <Stack.Screen name="team/[id]" options={{ title: "Team Details" }} />
        <Stack.Screen name="game/[id]" options={{ title: "Game Details" }} />
        <Stack.Screen name="(auth)/login" options={{ title: "Login", presentation: "modal" }} />
        <Stack.Screen name="(auth)/signup" options={{ title: "Sign Up", presentation: "modal" }} />
      </Stack>
    </QueryClientProvider>
  );
}